"""统一的 loguru 日志配置，以及第三方日志和原生输出的接入。"""

from __future__ import annotations

import argparse
import codecs
import ctypes
import io
import logging
import os
import re
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, NoReturn, Optional, TextIO, Union

from loguru import logger

# -------------------------------------------
# 配置常量
# -------------------------------------------
# fmt: off
LOG_ROTATION = "20 MB"             # 单个运行日志文件的轮转大小
LOG_RETENTION = "14 days"          # 轮转后日志文件的保留时长
PYTHON_STREAM_LINE_LIMIT = 8192    # Python 流无换行时强制输出的字符数
NATIVE_STREAM_LINE_LIMIT = 262144  # 原生输出无换行时强制输出的字符数
NATIVE_READ_BYTES = 65536          # 每次从原生输出管道读取的字节数
NATIVE_READER_JOIN_SECONDS = 5     # 恢复 stdout 后等待读取线程结束的秒数
# fmt: on

CONSOLE_FORMAT = "{time:HH:mm:ss} | {level:<7} | {extra[stage]} | {message}"
# 去掉第三方进度条等输出中的 ANSI 颜色和光标控制序列。
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
LINE_BREAK = re.compile(r"[\r\n]")


# -------------------------------------------
# 日志配置
# -------------------------------------------
class InterceptHandler(logging.Handler):
    """把第三方标准库日志记录转发给 Loguru，不单独输出。"""

    def emit(self, record: logging.LogRecord) -> None:
        """把一条标准库日志记录转发给 loguru，保留级别和来源。"""
        try:
            level: Union[str, int] = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        source = logger.bind(origin="stdlib", source_logger=record.name, source_file=record.pathname, source_line=record.lineno)
        source.opt(exception=record.exc_info).log(level, record.getMessage())


def intercept_standard_logging() -> None:
    """让所有标准库 logger 及 warning 都经由 loguru 输出。"""
    logging.basicConfig(handlers=[InterceptHandler()], level=logging.NOTSET, force=True)
    for existing in list(logging.root.manager.loggerDict.values()):
        if isinstance(existing, logging.Logger):
            existing.handlers.clear()
            existing.propagate = True
    logging.captureWarnings(True)


def configure_logging(
    output: Optional[Path] = None,
    run_id: str = "-",
    *,
    stage: str = "startup",
    view_id: str = "-",
    object_id: str = "-",
) -> None:
    """配置控制台日志，并在给定输出目录时追加 JSON Lines 文件日志。

    Args:
        output: 运行输出目录；日志写入 ``output/logs/<run_id>.log``。为 ``None`` 时只输出到控制台。
        run_id: 写入每条日志上下文的运行标识。
        stage: 默认阶段上下文。
        view_id: 默认视图上下文。
        object_id: 默认物体上下文。
    """
    logger.remove()
    context = {"run_id": run_id, "stage": stage, "view_id": view_id, "object_id": object_id, "origin": "application"}
    logger.configure(extra=context)
    logger.add(_original_stderr(), level="INFO", format=CONSOLE_FORMAT, diagnose=False, backtrace=False)
    if output is not None:
        log_dir = Path(output) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_dir / f"{run_id}.log"),
            level="DEBUG",
            rotation=LOG_ROTATION,
            retention=LOG_RETENTION,
            serialize=True,
            diagnose=False,
            backtrace=False,
        )
    intercept_standard_logging()


def _original_stderr() -> TextIO:
    """跳过 capture_external_output 安装的包装，返回真实的标准错误流。"""
    console = sys.stderr
    while isinstance(console, LoguruStream):
        console = console.original
    return console


class LoguruArgumentParser(argparse.ArgumentParser):
    """参数错误与帮助以外的退出信息也通过 loguru 输出的 ArgumentParser。"""

    def error(self, message: str) -> NoReturn:
        """记录参数错误并以退出码 2 结束。"""
        logger.error("{}: {}", self.prog, message)
        raise SystemExit(2)

    def exit(self, status: int = 0, message: Optional[str] = None) -> NoReturn:
        """记录退出信息后结束。"""
        if message:
            logger.log("ERROR" if status else "INFO", message.rstrip())
        raise SystemExit(status)


# -------------------------------------------
# 外部输出捕获
# -------------------------------------------
class LineSplitter:
    """按 CR/LF 切分字符流，并转发每个非空行。"""

    def __init__(self, emit: Callable[[str], Any], limit: int) -> None:
        self.emit = emit
        self.limit = limit
        self.pending = ""
        self.lock = threading.RLock()

    def write(self, value: str) -> None:
        """追加文本，转发其中完整的行；缓冲超过上限时整体转发。"""
        with self.lock:
            parts = LINE_BREAK.split(self.pending + value)
            self.pending = parts.pop()
            for part in parts:
                self._forward(part)
            if len(self.pending) >= self.limit:
                self._forward(self.pending)
                self.pending = ""

    def flush(self) -> None:
        """转发缓冲中尚未换行的剩余文本。"""
        with self.lock:
            if self.pending:
                self._forward(self.pending)
                self.pending = ""

    def _forward(self, value: str) -> None:
        value = ANSI_ESCAPE.sub("", value).strip()
        if value:
            self.emit(value)


class LoguruStream(io.TextIOBase):
    """替代 ``sys.stdout``/``sys.stderr``，把 Python 层写入转为 INFO 日志。"""

    def __init__(self, original: TextIO, name: str) -> None:
        self.original = original
        self.name = name
        self.lines = LineSplitter(logger.bind(origin="external_stream", stream=name).info, PYTHON_STREAM_LINE_LIMIT)

    @property
    def encoding(self) -> str:
        """沿用原始流的编码。"""
        return getattr(self.original, "encoding", None) or "utf-8"

    @property
    def buffer(self) -> Any:
        """原始流的二进制缓冲区。"""
        return self.original.buffer

    def fileno(self) -> int:
        """原始流的文件描述符。"""
        return self.original.fileno()

    def isatty(self) -> bool:
        """始终声明为非终端，避免第三方输出交互式控制序列。"""
        return False

    def writable(self) -> bool:
        """声明为可写流。"""
        return True

    def write(self, value: str) -> int:
        """按行转发文本并返回写入的字符数。

        Raises:
            TypeError: 写入的不是字符串。
        """
        if not isinstance(value, str):
            raise TypeError("LoguruStream.write requires a string")
        self.lines.write(value)
        return len(value)

    def flush(self) -> None:
        """转发尚未换行的剩余文本。"""
        self.lines.flush()


@contextmanager
def capture_native_stdout() -> Iterator[None]:
    """把 C/CUDA 扩展写入文件描述符 1 的原生输出转入 Loguru。

    模型与主程序在同一解释器中运行，否则原生 stdout 会混入 JSON 结果协议。
    文件描述符 2 保持不变，因为控制台日志 sink 写在那里。
    """
    try:
        sys.__stdout__.flush()
        saved = os.dup(1)
    except (AttributeError, OSError, ValueError):
        yield
        return
    read_end, write_end = os.pipe()
    os.dup2(write_end, 1)
    os.close(write_end)
    lines = LineSplitter(logger.bind(origin="native_output", stream="stdout").info, NATIVE_STREAM_LINE_LIMIT)
    reader = threading.Thread(target=_pump_native_output, args=(read_end, lines), name="native-stdout", daemon=True)
    reader.start()
    try:
        yield
    finally:
        _flush_stdout_buffers()
        os.dup2(saved, 1)
        os.close(saved)
        reader.join(timeout=NATIVE_READER_JOIN_SECONDS)
        if not reader.is_alive():
            os.close(read_end)


def _pump_native_output(read_end: int, lines: LineSplitter) -> None:
    """把管道中的原生输出逐块解码后交给 ``lines``，直到写端全部关闭。"""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    while True:
        block = os.read(read_end, NATIVE_READ_BYTES)
        if not block:
            break
        lines.write(decoder.decode(block))
    lines.write(decoder.decode(b"", final=True))
    lines.flush()


def _flush_stdout_buffers() -> None:
    """在恢复 fd 1 之前清空 Python 与 C 运行库的 stdout 缓冲。"""
    try:
        sys.__stdout__.flush()
    except (AttributeError, OSError, ValueError):
        pass
    try:
        ctypes.CDLL(None).fflush(None)
    except (OSError, AttributeError):
        pass


@contextmanager
def capture_external_output() -> Iterator[None]:
    """在上下文内把 Python 层 stdout/stderr 与原生 stdout 全部转入 loguru。"""
    stdout = LoguruStream(sys.stdout, "stdout")
    stderr = LoguruStream(sys.stderr, "stderr")
    with capture_native_stdout(), redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            yield
        finally:
            stdout.flush()
            stderr.flush()
