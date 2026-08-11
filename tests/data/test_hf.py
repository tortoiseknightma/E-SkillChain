import pytest

from skillchain.data._hf import download_file


class _Response:
    def __init__(self, *, status_code=200, content=b"", length=None):
        self.status_code = status_code
        self._content = content
        self.headers = {"Content-Length": str(length if length is not None else len(content))}

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield self._content

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_download_file_overwrites_partial_when_server_ignores_range(tmp_path, monkeypatch):
    destination = tmp_path / "data.bin"
    destination.write_bytes(b"abc")
    monkeypatch.setattr(
        "skillchain.data._hf.requests.head", lambda *a, **k: _Response(length=6)
    )
    monkeypatch.setattr(
        "skillchain.data._hf.requests.get",
        lambda *a, **k: _Response(status_code=200, content=b"abcdef"),
    )

    download_file("owner/repo", "data.bin", destination)

    assert destination.read_bytes() == b"abcdef"


def test_download_file_appends_only_partial_content_response(tmp_path, monkeypatch):
    destination = tmp_path / "data.bin"
    destination.write_bytes(b"abc")
    monkeypatch.setattr(
        "skillchain.data._hf.requests.head", lambda *a, **k: _Response(length=6)
    )
    monkeypatch.setattr(
        "skillchain.data._hf.requests.get",
        lambda *a, **k: _Response(status_code=206, content=b"def"),
    )

    download_file("owner/repo", "data.bin", destination)

    assert destination.read_bytes() == b"abcdef"


def test_download_file_rejects_local_file_larger_than_remote(tmp_path, monkeypatch):
    destination = tmp_path / "data.bin"
    destination.write_bytes(b"abcdefg")
    monkeypatch.setattr(
        "skillchain.data._hf.requests.head", lambda *a, **k: _Response(length=6)
    )

    with pytest.raises(RuntimeError, match="比远端文件更大"):
        download_file("owner/repo", "data.bin", destination)
