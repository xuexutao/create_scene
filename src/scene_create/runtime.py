from __future__ import annotations

import codecs
import fcntl
import os
import re
import selectors
import signal
import subprocess
import time
import uuid
from collections import deque
from contextlib import contextmanager
from pathlib import Path

from loguru import logger

from scene_create.io import atomic_json, file_digest, read_json
from scene_create.logging_utils import configure_logging as configure_logging
from scene_create.logging_utils import relay_worker_line


class WorkerError(RuntimeError):
    pass


@contextmanager
def output_lock(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process is writing to {output}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def stream_worker_output(process, worker_logger, timeout):
    deadline = time.monotonic() + timeout
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""
    tail = deque(maxlen=40)
    exited_at = None

    def emit(line):
        message = relay_worker_line(line, worker_logger)
        if message:
            tail.append(message[-12000:])

    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                emit(pending + decoder.decode(b"", final=True))
                raise subprocess.TimeoutExpired(process.args, timeout)
            for key, _ in selector.select(timeout=min(remaining, 0.1)):
                block = os.read(key.fileobj.fileno(), 65536)
                if not block:
                    pending += decoder.decode(b"", final=True)
                    emit(pending)
                    pending = ""
                    selector.unregister(key.fileobj)
                    continue
                parts = re.split(r"[\r\n]", pending + decoder.decode(block))
                pending = parts.pop()
                for part in parts:
                    emit(part)
                if len(pending) >= 262144:
                    emit(pending)
                    pending = ""
            if process.poll() is not None:
                if exited_at is None:
                    exited_at = time.monotonic()
                elif selector.get_map() and time.monotonic() - exited_at > 5:
                    raise WorkerError(
                        "Worker exited but a descendant kept its output pipe open"
                    )
        code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
    return code, "\n".join(tail)[-12000:]


def stop_worker(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


class WorkerRunner:
    def __init__(self, config, output, run_id=None):
        self.config = config
        self.output = Path(output)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.sequence = 0

    def call(self, operation, payload, destination, probe=False):
        self.sequence += 1
        job = Path(destination) / f"worker_{operation}_{self.sequence:04d}"
        job.mkdir(parents=True, exist_ok=True)
        request_path, response_path = job / "request.json", job / "response.json"
        request = {
            "operation": operation,
            "root": self.config["root"],
            "config": self.config["models"][operation],
            "output_dir": str(Path(destination).resolve()),
            **payload,
        }
        views = payload.get("views", [])
        context = {
            "run_id": self.run_id,
            "stage": operation,
            "object_id": payload.get("object", {}).get("object_id", "-"),
            "view_id": views[0].get("view_id", "-") if len(views) == 1 else "-",
        }
        command = [
            request["config"]["python"],
            "-m",
            "scene_create.worker",
            "--request",
            str(request_path),
            "--response",
            str(response_path),
        ]
        if probe:
            command.append("--probe")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = (
            str(Path(self.config["root"]) / "src")
            + os.pathsep
            + environment.get("PYTHONPATH", "")
        )
        environment["PYTHONUNBUFFERED"] = "1"
        timeout = (
            min(60, self.config["runtime"]["worker_timeout"])
            if probe
            else self.config["runtime"]["worker_timeout"]
        )
        attempts = 1 if probe else 1 + self.config["runtime"]["retries"]
        for attempt in range(attempts):
            response_path.unlink(missing_ok=True)
            log_path = job / f"worker_{attempt}.log"
            worker_id = f"{self.run_id}-{self.sequence}-{attempt}"
            request["logging_context"] = {**context, "worker_id": worker_id}
            atomic_json(request_path, request)
            log = logger.bind(**request["logging_context"])
            sink = logger.add(
                str(log_path),
                level="DEBUG",
                serialize=True,
                mode="w",
                rotation="20 MB",
                retention="14 days",
                diagnose=False,
                backtrace=False,
                filter=lambda record, key=worker_id: record["extra"].get("worker_id")
                == key,
            )
            try:
                log.info(
                    "Launching {} worker, attempt {}/{}",
                    operation,
                    attempt + 1,
                    attempts,
                )
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=self.config["root"],
                        env=environment,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                except OSError as error:
                    log.exception("Cannot start worker: {}", error)
                    raise WorkerError(
                        f"Cannot start {operation} worker: {error}"
                    ) from error
                try:
                    code, tail = stream_worker_output(process, log, timeout)
                except subprocess.TimeoutExpired as error:
                    stop_worker(process)
                    log.error("{} worker timed out after {}s", operation, timeout)
                    raise WorkerError(
                        f"{operation} timed out after {timeout}s; see {log_path}"
                    ) from error
                except KeyboardInterrupt:
                    stop_worker(process)
                    log.warning("{} worker interrupted", operation)
                    raise
                except Exception:
                    stop_worker(process)
                    log.exception("Worker output processing failed")
                    raise
                finally:
                    process.stdout.close()
                response = read_json(response_path) if response_path.exists() else {}
                if code == 0 and response.get("status") == "success":
                    return response["result"]
                error = response.get("error", f"process exit {code}; {tail[-1500:]}")
                log.error("Worker exited with status {}: {}", code, error)
                if attempt + 1 == attempts:
                    raise WorkerError(f"{operation}: {error}. Worker log: {log_path}")
                log.warning("Worker failed; retry {}/{}", attempt + 1, attempts - 1)
            finally:
                logger.remove(sink)
            time.sleep(min(2**attempt, 10))
        raise WorkerError("Worker did not return a result")


class StageStore:
    def __init__(self, output, run_id, resume=False):
        self.output = Path(output)
        self.path = self.output / "run_manifest.json"
        if self.path.exists() and not resume:
            raise ValueError(
                f"{self.path} exists; pass --resume or choose another output directory"
            )
        self.data = (
            read_json(self.path)
            if self.path.exists()
            else {"schema_version": 1, "run_id": run_id, "stages": {}}
        )
        self.data["last_run_id"] = run_id
        self.data["status"] = "running"
        atomic_json(self.path, self.data)

    def run(self, name, digest, function):
        previous = self.data["stages"].get(name, {})
        cached = (
            previous.get("status") == "completed"
            and previous.get("fingerprint") == digest
        )
        if cached:
            for artifact in previous.get("artifacts", []):
                path = Path(artifact["path"])
                if not path.is_file() or file_digest(path) != artifact["sha256"]:
                    cached = False
                    break
        if cached:
            logger.bind(stage=name).info("Using verified cached result")
            return previous["result"]
        directory = self.output / "intermediate" / name / digest[:16]
        directory.mkdir(parents=True, exist_ok=True)
        entry = {"status": "running", "fingerprint": digest, "started_at": time.time()}
        self.data["stages"][name] = entry
        atomic_json(self.path, self.data)
        started = time.monotonic()
        log = logger.bind(stage=name)
        log.info("Starting stage")
        try:
            result, artifacts = function(directory)
            result_path = directory / "result.json"
            atomic_json(result_path, result)
            files = sorted(
                {str(Path(path).resolve()) for path in [*artifacts, result_path]}
            )
            entry.update(
                status="partial" if result.get("failures") else "completed",
                result=read_json(result_path),
                artifacts=[
                    {"path": path, "sha256": file_digest(path)} for path in files
                ],
                elapsed_seconds=time.monotonic() - started,
            )
            atomic_json(self.path, self.data)
            log.info("Completed in {:.2f}s", entry["elapsed_seconds"])
            return entry["result"]
        except Exception as error:
            entry.update(
                status="failed",
                error_type=type(error).__name__,
                error=str(error),
                elapsed_seconds=time.monotonic() - started,
            )
            self.data["status"] = "failed"
            atomic_json(self.path, self.data)
            log.exception("Stage failed: {}", error)
            raise

    def finish(self, status):
        self.data["status"] = status
        self.data["finished_at"] = time.time()
        atomic_json(self.path, self.data)
