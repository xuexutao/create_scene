from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sys
import threading
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

from loguru import logger

ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class InterceptHandler(logging.Handler):
    """Forward third-party stdlib records to Loguru; never emit a separate log."""

    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.bind(
            origin="stdlib",
            source_logger=record.name,
            source_file=record.pathname,
            source_line=record.lineno,
        ).opt(exception=record.exc_info).log(level, record.getMessage())


def intercept_standard_logging():
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.NOTSET, force=True)
    for existing in list(logging.root.manager.loggerDict.values()):
        if isinstance(existing, logging.Logger):
            existing.handlers.clear()
            existing.propagate = True
    logging.captureWarnings(True)


def configure_logging(
    output=None,
    run_id="-",
    *,
    stage="startup",
    view_id="-",
    object_id="-",
    worker_id="-",
    serialized_console=False,
):
    logger.remove()
    logger.configure(
        extra={
            "run_id": run_id,
            "stage": stage,
            "view_id": view_id,
            "object_id": object_id,
            "worker_id": worker_id,
            "origin": "application",
        }
    )
    console = sys.stderr
    while isinstance(console, LoguruStream):
        console = console.original
    logger.add(
        console,
        level="DEBUG" if serialized_console else "INFO",
        format="{message}"
        if serialized_console
        else "{time:HH:mm:ss} | {level:<7} | {extra[stage]} | {message}",
        serialize=serialized_console,
        diagnose=False,
        backtrace=False,
    )
    if output is not None:
        log_dir = Path(output) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_dir / f"{run_id}.log"),
            level="DEBUG",
            rotation="20 MB",
            retention="14 days",
            serialize=True,
            diagnose=False,
            backtrace=False,
        )
    intercept_standard_logging()


class LoguruArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        logger.error("{}: {}", self.prog, message)
        raise SystemExit(2)

    def exit(self, status=0, message=None):
        if message:
            logger.log("ERROR" if status else "INFO", message.rstrip())
        raise SystemExit(status)


class LoguruStream(io.TextIOBase):
    def __init__(self, original, name):
        self.original = original
        self.name = name
        self.pending = ""
        self.lock = threading.RLock()

    @property
    def encoding(self):
        return getattr(self.original, "encoding", None) or "utf-8"

    @property
    def buffer(self):
        return self.original.buffer

    def fileno(self):
        return self.original.fileno()

    def isatty(self):
        return False

    def writable(self):
        return True

    def _emit(self, value):
        value = ANSI_ESCAPE.sub("", value).strip()
        if value:
            logger.bind(origin="external_stream", stream=self.name).info("{}", value)

    def write(self, value):
        if not isinstance(value, str):
            raise TypeError("LoguruStream.write requires a string")
        with self.lock:
            parts = re.split(r"[\r\n]", self.pending + value)
            self.pending = parts.pop()
            for part in parts:
                self._emit(part)
            if len(self.pending) >= 8192:
                self._emit(self.pending)
                self.pending = ""
        return len(value)

    def flush(self):
        with self.lock:
            if self.pending:
                self._emit(self.pending)
                self.pending = ""


@contextmanager
def capture_external_output():
    stdout = LoguruStream(sys.stdout, "stdout")
    stderr = LoguruStream(sys.stderr, "stderr")
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            yield
        finally:
            stdout.flush()
            stderr.flush()


def relay_worker_line(line, worker_logger):
    """Preserve structured worker levels; native stdout/stderr becomes an INFO record."""
    line = ANSI_ESCAPE.sub("", line).strip()
    if not line:
        return ""
    try:
        payload = json.loads(line)
    except (ValueError, TypeError):
        payload = None
    record = payload.get("record") if isinstance(payload, dict) else None
    if (
        isinstance(record, dict)
        and isinstance(record.get("level"), dict)
        and isinstance(record.get("message"), str)
    ):
        level = record["level"].get("name", "INFO")
        try:
            logger.level(level)
        except (ValueError, TypeError):
            level = "INFO"
        extra = record.get("extra")
        extra = extra if isinstance(extra, dict) else {}
        message = record["message"]
        if record.get("exception") and isinstance(payload.get("text"), str):
            message = payload["text"].rstrip()
        worker_logger.bind(
            origin=extra.get("origin", "worker"),
            stream=extra.get("stream"),
            source_logger=extra.get("source_logger"),
            worker_timestamp=record.get("time", {}).get("repr")
            if isinstance(record.get("time"), dict)
            else None,
            worker_exception=record.get("exception"),
        ).log(level, "{}", message)
        return message
    worker_logger.bind(origin="native_worker_output").info("{}", line)
    return line
