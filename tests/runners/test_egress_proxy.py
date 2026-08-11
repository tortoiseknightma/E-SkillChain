from __future__ import annotations

import pytest

from skillchain.runners.egress_proxy import _parse_connect_target


@pytest.mark.parametrize(
    ("request_line", "expected"),
    [
        (b"CONNECT api.example.com:443 HTTP/1.1", ("api.example.com", 443)),
        (b"CONNECT API.EXAMPLE.COM:443 HTTP/1.0", ("api.example.com", 443)),
    ],
)
def test_parse_connect_target(request_line, expected):
    assert _parse_connect_target(request_line) == expected


@pytest.mark.parametrize(
    "request_line",
    [
        b"GET https://api.example.com HTTP/1.1",
        b"CONNECT 127.0.0.1:443 HTTP/1.1",
        b"CONNECT [::1]:443 HTTP/1.1",
        b"CONNECT api.example.com:443 HTTP/2",
        b"CONNECT api.example.com:not-a-port HTTP/1.1",
        b"CONNECT api.example.com.:443 HTTP/1.1",
    ],
)
def test_parse_connect_target_rejects_bypass_shapes(request_line):
    with pytest.raises(ValueError):
        _parse_connect_target(request_line)
