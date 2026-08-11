"""Exact-host HTTPS CONNECT proxy for the formal authoring sandbox.

The authoring container is attached only to a Docker ``--internal`` network.
This proxy is the sole dual-homed peer and permits TLS tunnels to one public
provider hostname on port 443.  It never receives provider credentials and it
cannot inspect or persist request bodies.
"""

from __future__ import annotations

import ipaddress
import json
import os
import selectors
import socket
import socketserver
import sys
from typing import BinaryIO

_MAX_HEADER_BYTES = 8192
_CONNECT_TIMEOUT_SECONDS = 10
_IDLE_TIMEOUT_SECONDS = 120


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "")
    if not value or value != value.strip():
        raise RuntimeError(f"missing or invalid {name}")
    return value


def _parse_connect_target(request_line: bytes) -> tuple[str, int]:
    try:
        method, authority, version = request_line.decode("ascii").split(" ")
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("malformed proxy request line") from error
    if method != "CONNECT" or version not in {"HTTP/1.0", "HTTP/1.1"}:
        raise ValueError("only HTTPS CONNECT is supported")
    if authority.count(":") != 1:
        raise ValueError("CONNECT target must be hostname:port")
    hostname, port_text = authority.rsplit(":", 1)
    hostname = hostname.casefold()
    if (
        not hostname
        or hostname.endswith(".")
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-."
            for character in hostname
        )
    ):
        raise ValueError("CONNECT hostname is invalid")
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        raise ValueError("IP-literal CONNECT targets are forbidden")
    try:
        port = int(port_text)
    except ValueError as error:
        raise ValueError("CONNECT port is invalid") from error
    return hostname, port


def _resolve_public(hostname: str, port: int) -> tuple[int, int, int, tuple]:
    addresses = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    if not addresses:
        raise OSError("provider hostname did not resolve")
    for _family, _socktype, _proto, _canonname, sockaddr in addresses:
        if not ipaddress.ip_address(sockaddr[0]).is_global:
            raise OSError("provider hostname resolved to a non-public address")
    family, socktype, proto, _canonname, sockaddr = addresses[0]
    return family, socktype, proto, sockaddr


def _write_response(stream: BinaryIO, status: int, reason: str) -> None:
    stream.write(
        f"HTTP/1.1 {status} {reason}\r\nConnection: close\r\n\r\n".encode("ascii")
    )
    stream.flush()


class _ConnectHandler(socketserver.StreamRequestHandler):
    server: "_ProxyServer"

    def handle(self) -> None:
        self.connection.settimeout(_CONNECT_TIMEOUT_SECONDS)
        request = self.rfile.readline(_MAX_HEADER_BYTES + 1)
        if len(request) > _MAX_HEADER_BYTES or not request.endswith(b"\r\n"):
            _write_response(self.wfile, 400, "Bad Request")
            return
        header_bytes = len(request)
        while True:
            line = self.rfile.readline(_MAX_HEADER_BYTES + 1)
            header_bytes += len(line)
            if header_bytes > _MAX_HEADER_BYTES or not line:
                _write_response(self.wfile, 400, "Bad Request")
                return
            if line == b"\r\n":
                break
        try:
            hostname, port = _parse_connect_target(request.rstrip(b"\r\n"))
        except ValueError:
            self.server.audit("invalid", 0, False, "malformed")
            _write_response(self.wfile, 400, "Bad Request")
            return
        allowed = (
            hostname == self.server.allowed_host and port == self.server.allowed_port
        )
        if not allowed:
            self.server.audit(hostname, port, False, "not_allowlisted")
            _write_response(self.wfile, 403, "Forbidden")
            return
        try:
            family, socktype, proto, sockaddr = _resolve_public(hostname, port)
            upstream = socket.socket(family, socktype, proto)
            upstream.settimeout(_CONNECT_TIMEOUT_SECONDS)
            upstream.connect(sockaddr)
        except OSError:
            self.server.audit(hostname, port, False, "upstream_unavailable")
            _write_response(self.wfile, 502, "Bad Gateway")
            return
        self.server.audit(hostname, port, True, "allowlisted")
        _write_response(self.wfile, 200, "Connection Established")
        self.connection.settimeout(_IDLE_TIMEOUT_SECONDS)
        upstream.settimeout(_IDLE_TIMEOUT_SECONDS)
        selector = selectors.DefaultSelector()
        selector.register(self.connection, selectors.EVENT_READ, upstream)
        selector.register(upstream, selectors.EVENT_READ, self.connection)
        try:
            while True:
                events = selector.select(_IDLE_TIMEOUT_SECONDS)
                if not events:
                    break
                for key, _mask in events:
                    chunk = key.fileobj.recv(65536)
                    if not chunk:
                        return
                    key.data.sendall(chunk)
        finally:
            selector.close()
            upstream.close()


class _ProxyServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int], allowed_host: str, allowed_port: int):
        self.allowed_host = allowed_host.casefold()
        self.allowed_port = allowed_port
        super().__init__(address, _ConnectHandler)

    @staticmethod
    def audit(hostname: str, port: int, allowed: bool, reason: str) -> None:
        print(
            json.dumps(
                {
                    "allowed": allowed,
                    "hostname": hostname,
                    "port": port,
                    "reason": reason,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )


def main() -> int:
    allowed_host = _required_environment("SKILLCHAIN_ALLOWED_HOST").casefold()
    if allowed_host == "localhost":
        raise RuntimeError("localhost cannot be an allowed provider")
    try:
        ipaddress.ip_address(allowed_host)
    except ValueError:
        pass
    else:
        raise RuntimeError("allowed provider must be a hostname")
    allowed_port = int(_required_environment("SKILLCHAIN_ALLOWED_PORT"))
    if allowed_port != 443:
        raise RuntimeError("only provider port 443 may be allowed")
    server = _ProxyServer(("0.0.0.0", 3128), allowed_host, allowed_port)
    print(
        json.dumps(
            {
                "event": "ready",
                "hostname": allowed_host,
                "listen_port": 3128,
                "port": allowed_port,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        file=sys.stdout,
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":  # pragma: no cover - container entry point
    raise SystemExit(main())
