from __future__ import annotations

import argparse
import http.client
import json
import logging
import os
import signal
import stat
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO


LOGGER = logging.getLogger("qwen-bridge")
MAX_REQUEST_BODY = 2 * 1024 * 1024
SECRET_MAX_BYTES = 8192
ALLOWED_ROUTES = {
    ("GET", "/v1/models"),
    ("POST", "/v1/chat/completions"),
}
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
FORWARDED_REQUEST_HEADERS = {
    "accept",
    "content-type",
    "x-request-id",
}


@dataclass(frozen=True)
class BridgeConfig:
    listen_host: str
    listen_port: int
    upstream_host: str
    upstream_port: int
    caller_key: str
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 1860.0
    max_request_body: int = MAX_REQUEST_BODY


def read_caller_key(path: str | os.PathLike[str]) -> str:
    secret_path = Path(path)
    info = secret_path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("caller key path is not a regular file")
    if info.st_size <= 0 or info.st_size > SECRET_MAX_BYTES:
        raise ValueError("caller key file has an invalid size")
    value = secret_path.read_text(encoding="utf-8").strip()
    if not value or "\n" in value or "\r" in value:
        raise ValueError("caller key must be one non-empty line")
    return value


def load_config() -> BridgeConfig:
    max_request_body = int(
        os.environ.get("QWEN_BRIDGE_MAX_REQUEST_BODY", str(MAX_REQUEST_BODY))
    )
    if not 0 < max_request_body <= MAX_REQUEST_BODY:
        raise ValueError(
            f"QWEN_BRIDGE_MAX_REQUEST_BODY must be between 1 and {MAX_REQUEST_BODY}"
        )
    config = BridgeConfig(
        listen_host=os.environ.get("QWEN_BRIDGE_LISTEN_HOST", "0.0.0.0"),
        listen_port=int(os.environ.get("QWEN_BRIDGE_LISTEN_PORT", "8000")),
        upstream_host=os.environ.get("QWEN_BRIDGE_UPSTREAM_HOST", "gateway"),
        upstream_port=int(os.environ.get("QWEN_BRIDGE_UPSTREAM_PORT", "8000")),
        caller_key=read_caller_key(
            os.environ.get(
                "QWEN_BRIDGE_CALLER_KEY_FILE",
                "<redacted-secret-path>",
            )
        ),
        connect_timeout_seconds=float(
            os.environ.get("QWEN_BRIDGE_CONNECT_TIMEOUT_SECONDS", "10")
        ),
        read_timeout_seconds=float(
            os.environ.get("QWEN_BRIDGE_READ_TIMEOUT_SECONDS", "1860")
        ),
        max_request_body=max_request_body,
    )
    if not 0 < config.listen_port <= 65535 or not 0 < config.upstream_port <= 65535:
        raise ValueError("listen and upstream ports must be between 1 and 65535")
    if config.connect_timeout_seconds <= 0 or config.read_timeout_seconds <= 0:
        raise ValueError("timeouts must be positive")
    return config


class QwenBridgeServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, config: BridgeConfig):
        self.bridge_config = config
        super().__init__((config.listen_host, config.listen_port), QwenBridgeHandler)


class QwenBridgeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "qwen-bridge"
    sys_version = ""

    @property
    def config(self) -> BridgeConfig:
        server = self.server
        if not isinstance(server, QwenBridgeServer):
            raise RuntimeError("invalid bridge server")
        return server.bridge_config

    def log_message(self, format: str, *args: object) -> None:
        LOGGER.info("client=%s %s", self.client_address[0], format % args)

    def _send_json_error(self, status_code: int, code: str) -> None:
        body = json.dumps({"error": {"code": code}}, separators=(",", ":")).encode(
            "utf-8"
        )
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        if self.command != "HEAD":
            self.wfile.write(body)

    def _read_post_body(self) -> bytes | None:
        if self.headers.get("Transfer-Encoding") is not None:
            self._send_json_error(400, "chunked_request_not_supported")
            return None
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            self._send_json_error(411, "content_length_required")
            return None
        try:
            length = int(raw_length, 10)
        except ValueError:
            self._send_json_error(400, "invalid_content_length")
            return None
        if length < 0:
            self._send_json_error(400, "invalid_content_length")
            return None
        if length > self.config.max_request_body:
            self._send_json_error(413, "request_body_too_large")
            return None
        body = self.rfile.read(length)
        if len(body) != length:
            self._send_json_error(400, "incomplete_request_body")
            return None
        try:
            decoded = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json_error(400, "invalid_json")
            return None
        if not isinstance(decoded, dict):
            self._send_json_error(400, "json_object_required")
            return None
        return body

    def _request_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.config.caller_key}",
            "Connection": "close",
        }
        for name in FORWARDED_REQUEST_HEADERS:
            value = self.headers.get(name)
            if value is not None:
                headers[name] = value
        return headers

    def _relay_response_body(self, source: BinaryIO) -> None:
        read1 = getattr(source, "read1", source.read)
        while True:
            chunk = read1(64 * 1024)
            if not chunk:
                return
            self.wfile.write(chunk)
            self.wfile.flush()

    def _proxy(self, body: bytes | None) -> None:
        upstream = http.client.HTTPConnection(
            self.config.upstream_host,
            self.config.upstream_port,
            timeout=self.config.connect_timeout_seconds,
        )
        response: http.client.HTTPResponse | None = None
        response_started = False
        try:
            upstream.request(
                self.command,
                self.path,
                body=body,
                headers=self._request_headers(),
            )
            if upstream.sock is not None:
                upstream.sock.settimeout(self.config.read_timeout_seconds)
            response = upstream.getresponse()
            self.send_response(response.status, response.reason)
            for name, value in response.getheaders():
                lowered = name.lower()
                if lowered in HOP_BY_HOP_HEADERS:
                    continue
                self.send_header(name, value)
            self.send_header("Connection", "close")
            self.end_headers()
            response_started = True
            self.close_connection = True
            self._relay_response_body(response)
        except (BrokenPipeError, ConnectionResetError):
            LOGGER.info("downstream_disconnected path=%s", self.path)
        except (OSError, http.client.HTTPException) as exc:
            LOGGER.warning("upstream_error path=%s type=%s", self.path, type(exc).__name__)
            if not response_started:
                self._send_json_error(502, "upstream_unavailable")
            else:
                self.close_connection = True
        finally:
            if response is not None:
                response.close()
            upstream.close()

    def do_GET(self) -> None:
        if ("GET", self.path) not in ALLOWED_ROUTES:
            self._send_json_error(404, "route_not_found")
            return
        self._proxy(None)

    def do_POST(self) -> None:
        if ("POST", self.path) not in ALLOWED_ROUTES:
            self._send_json_error(404, "route_not_found")
            return
        body = self._read_post_body()
        if body is not None:
            self._proxy(body)

    def do_HEAD(self) -> None:
        self._send_json_error(405, "method_not_allowed")

    def do_PUT(self) -> None:
        self._send_json_error(405, "method_not_allowed")

    def do_PATCH(self) -> None:
        self._send_json_error(405, "method_not_allowed")

    def do_DELETE(self) -> None:
        self._send_json_error(405, "method_not_allowed")

    def do_OPTIONS(self) -> None:
        self._send_json_error(405, "method_not_allowed")


def create_server(config: BridgeConfig) -> QwenBridgeServer:
    return QwenBridgeServer(config)


def main() -> int:
    parser = argparse.ArgumentParser(description="Loopback Qwen gateway bridge")
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    config = load_config()
    if args.check_config:
        return 0
    server = create_server(config)
    shutdown_started = threading.Event()

    def stop(_signum: int, _frame: object) -> None:
        if shutdown_started.is_set():
            return
        shutdown_started.set()
        # BaseServer.shutdown() must run outside the serve_forever() thread.
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    LOGGER.info(
        "listening host=%s port=%d upstream=%s:%d",
        config.listen_host,
        config.listen_port,
        config.upstream_host,
        config.upstream_port,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
