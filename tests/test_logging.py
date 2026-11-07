import ast
import io
import json
import logging
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from loguru import logger
from PIL import Image

from scene_create import cli
from scene_create.config import load_config
from scene_create.io import read_json
from scene_create.logging_utils import (
    capture_external_output,
    configure_logging,
    relay_worker_line,
)
from scene_create.runtime import (
    WorkerError,
    WorkerRunner,
    stop_worker,
    stream_worker_output,
)


@pytest.fixture(autouse=True)
def restore_logging_state():
    root_handlers = list(logging.root.handlers)
    root_level = logging.root.level
    named = {
        name: (list(value.handlers), value.propagate, value.level)
        for name, value in list(logging.root.manager.loggerDict.items())
        if isinstance(value, logging.Logger)
    }
    showwarning = warnings.showwarning
    yield
    logger.remove()
    logging.captureWarnings(False)
    warnings.showwarning = showwarning
    logging.root.handlers = root_handlers
    logging.root.setLevel(root_level)
    for name, (handlers, propagate, level) in named.items():
        restored = logging.getLogger(name)
        restored.handlers = handlers
        restored.propagate = propagate
        restored.setLevel(level)


@pytest.fixture
def records(tmp_path):
    configure_logging(tmp_path, "test-run", stage="test")
    result = []
    logger.add(
        lambda message: result.append(message.record), level="DEBUG", diagnose=False
    )
    return result


def test_third_party_stdout_stderr_and_stdlib_use_loguru(records, capsys):
    vendor = logging.getLogger("vendor.test_logging")
    vendor.setLevel(logging.DEBUG)
    with capture_external_output():
        vendor.warning("vendor warning %s", "kept")
        sys.stdout.write("fragment ")
        sys.stdout.write("completed\n")
        sys.stderr.write("\x1b[32mprogress\x1b[0m\r")
        print("vendor print")
        warnings.warn("vendor Python warning", UserWarning)
        logger.info("application log")
        sys.stdout.write("no trailing newline")
    by_message = {record["message"]: record for record in records}
    assert by_message["vendor warning kept"]["level"].name == "WARNING"
    assert by_message["vendor warning kept"]["extra"]["source_logger"] == vendor.name
    assert by_message["fragment completed"]["extra"]["stream"] == "stdout"
    assert by_message["progress"]["extra"]["stream"] == "stderr"
    assert by_message["vendor print"]["extra"]["origin"] == "external_stream"
    assert "no trailing newline" in by_message
    assert (
        len([record for record in records if record["message"] == "application log"])
        == 1
    )
    assert any(
        record["extra"].get("source_logger") == "py.warnings"
        and "vendor Python warning" in record["message"]
        for record in records
    )
    assert capsys.readouterr().out == ""


def test_existing_standard_handlers_are_not_duplicated():
    vendor = logging.getLogger("vendor.previous_handler")
    old_output = io.StringIO()
    vendor.handlers = [logging.StreamHandler(old_output)]
    vendor.propagate = False
    configure_logging(stage="test")
    messages = []
    logger.add(lambda message: messages.append(message.record), diagnose=False)
    vendor.warning("one event")
    assert old_output.getvalue() == ""
    assert (
        len([message for message in messages if message["message"] == "one event"]) == 1
    )


def test_capture_restores_streams_on_exception(records):
    original_stdout, original_stderr = sys.stdout, sys.stderr
    with pytest.raises(ValueError):
        with capture_external_output():
            sys.stderr.write("last vendor detail")
            raise ValueError("expected")
    assert sys.stdout is original_stdout
    assert sys.stderr is original_stderr
    assert any(record["message"] == "last vendor detail" for record in records)


def test_worker_relay_keeps_levels_context_and_traceback(records):
    payload = {
        "text": "worker failure\nTraceback: retained frame\nValueError: failed\n",
        "record": {
            "message": "worker failure",
            "level": {"name": "ERROR", "no": 40},
            "exception": {"type": "ValueError", "value": "failed", "traceback": True},
            "extra": {"origin": "stdlib", "source_logger": "vendor.module"},
            "time": {"repr": "worker-time"},
        },
    }
    log = logger.bind(stage="trellis", object_id="chair_001", worker_id="job-1")
    relay_worker_line(json.dumps(payload), log)
    record = records[-1]
    assert record["level"].name == "ERROR"
    assert "retained frame" in record["message"]
    assert record["extra"]["object_id"] == "chair_001"
    assert record["extra"]["stage"] == "trellis"
    assert record["extra"]["worker_timestamp"] == "worker-time"
    relay_worker_line("{not a logging record}", log)
    assert records[-1]["extra"]["origin"] == "native_worker_output"


def test_native_and_structured_subprocess_logs_are_streamed(records):
    command = [
        sys.executable,
        "-c",
        "import os,sys; from loguru import logger; logger.remove(); logger.add(sys.stderr, serialize=True, format='{message}'); logger.warning('preserved warning'); os.write(1, '原生输出'.encode('utf-8') + bytes([13]) + '最后一行'.encode('utf-8'))",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        code, tail = stream_worker_output(
            process, logger.bind(worker_id="native-test"), 15
        )
    finally:
        if process.poll() is None:
            stop_worker(process)
        process.stdout.close()
    assert code == 0
    assert "原生输出" in tail and "最后一行" in tail
    warning = next(
        record for record in records if record["message"] == "preserved warning"
    )
    assert warning["level"].name == "WARNING"
    assert any(record["message"] == "最后一行" for record in records)


def test_worker_progress_is_forwarded_before_process_exit(records):
    command = [
        sys.executable,
        "-c",
        "import os,sys; os.write(1, b'waiting-for-log' + bytes([10])); sys.stdin.readline()",
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    def acknowledge(message):
        if message.record["message"] == "waiting-for-log":
            process.stdin.write(b"continue\n")
            process.stdin.flush()

    sink = logger.add(acknowledge)
    try:
        code, _ = stream_worker_output(process, logger.bind(worker_id="live-test"), 10)
    finally:
        logger.remove(sink)
        if process.poll() is None:
            stop_worker(process)
        process.stdin.close()
        process.stdout.close()
    assert code == 0


def test_timeout_keeps_unterminated_worker_message(records):
    command = [
        sys.executable,
        "-c",
        "import os,time; os.write(1, b'before-timeout'); time.sleep(10)",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            stream_worker_output(process, logger.bind(worker_id="timeout-test"), 0.5)
    finally:
        stop_worker(process)
        process.stdout.close()
    assert any(record["message"] == "before-timeout" for record in records)


def test_worker_failure_is_logged_and_response_is_preserved(tmp_path, records):
    runner = WorkerRunner(load_config(), tmp_path, run_id="test-run")
    with pytest.raises(WorkerError, match="model_path"):
        runner.call("sam3", {"object": {"object_id": "chair_001"}}, tmp_path / "job")
    response_path = next((tmp_path / "job").rglob("response.json"))
    assert read_json(response_path)["status"] == "failed"
    worker_log = next((tmp_path / "job").rglob("worker_0.log"))
    events = [
        json.loads(line)["record"] for line in worker_log.read_text().splitlines()
    ]
    assert events and all(event["extra"]["run_id"] == "test-run" for event in events)
    assert all(event["extra"]["object_id"] == "chair_001" for event in events)
    assert any(
        event["level"]["name"] == "ERROR" and "FileNotFoundError" in event["message"]
        for event in events
    )
    scene_log = tmp_path / "logs" / "test-run.log"
    assert "FileNotFoundError" in scene_log.read_text()


def test_cli_result_stdout_is_pure_json(tmp_path, capsys):
    image = tmp_path / "image.png"
    Image.new("RGB", (24, 24), "white").save(image)
    code = cli.main(
        [
            "run",
            "--image",
            str(image),
            "--through",
            "ingest",
            "--output",
            str(tmp_path / "scene"),
        ]
    )
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert code == 0 and result["status"] == "completed"
    assert "INFO" in captured.err
    assert "INFO" not in captured.out


def test_doctor_output_remains_json_with_external_noise(monkeypatch, capsys):
    from scene_create import doctor

    def report(config):
        sys.stdout.write("dependency diagnostic\n")
        logging.getLogger("doctor.vendor").warning("dependency unavailable")
        return {"ready": False, "models": {}}

    monkeypatch.setattr(doctor, "diagnose", report)
    assert cli.main(["doctor"]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"ready": False, "models": {}}
    assert "dependency diagnostic" in captured.err
    assert "WARNING" in captured.err


def test_result_serializer_is_not_diagnostic_logging(capsys):
    cli.emit_result({"points": np.array([1, 2]), "path": Path("asset.ply")})
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"points": [1, 2], "path": "asset.ply"}
    assert captured.err == ""


def test_cli_failure_uses_loguru(tmp_path, capsys):
    code = cli.main(["run", "--image", str(tmp_path / "missing.png")])
    captured = capsys.readouterr()
    assert code == 2
    assert not captured.out
    assert "ERROR" in captured.err and "Command failed" in captured.err
    assert "FileNotFoundError" in captured.err


def test_cli_argument_errors_use_loguru(capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["run"])
    captured = capsys.readouterr()
    assert exit_info.value.code == 2
    assert not captured.out
    assert "ERROR" in captured.err and "required" in captured.err


def test_cli_keyboard_interrupt_uses_loguru(monkeypatch, capsys):
    def interrupt():
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "make_parser", interrupt)
    assert cli.main([]) == 130
    captured = capsys.readouterr()
    assert not captured.out
    assert "WARNING" in captured.err and "--resume" in captured.err


def test_owned_sources_do_not_use_print_or_traceback_diagnostics():
    root = Path(__file__).resolve().parents[1]
    for path in [
        root / "main.py",
        *sorted((root / "src" / "scene_create").rglob("*.py")),
    ]:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                assert not (
                    isinstance(node.func, ast.Name) and node.func.id == "print"
                ), str(path)
                assert not (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "traceback"
                    and node.func.attr.startswith("print")
                ), str(path)
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module]
                )
                if "logging" in modules:
                    assert path.name == "logging_utils.py", str(path)
