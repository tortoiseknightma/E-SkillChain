"""hf-mirror.com 文件下载工具（断点续传）。各数据源模块共用。"""

from pathlib import Path

import requests

MIRROR = "https://hf-mirror.com/datasets"


def download_file(repo: str, filename: str, dest: Path) -> Path:
    """从 hf-mirror 下载数据集文件，支持断点续传。已完整则跳过。"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{MIRROR}/{repo}/resolve/main/{filename}"
    head = requests.head(url, timeout=30, allow_redirects=True)
    head.raise_for_status()
    remote_size = int(head.headers["Content-Length"])
    have = dest.stat().st_size if dest.exists() else 0
    if have == remote_size:
        print(f"[skip] {dest.name} 已完整 ({have/1e6:.0f} MB)")
        return dest
    if have > remote_size:
        raise RuntimeError(f"{dest} 比远端文件更大，请人工检查后重试")
    print(f"[get ] {repo}/{filename} {have/1e6:.0f}/{remote_size/1e6:.0f} MB")
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        mode = "ab" if have and r.status_code == 206 else "wb"
        with open(dest, mode) as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    actual = dest.stat().st_size
    if actual != remote_size:
        raise RuntimeError(
            f"{dest.name} 下载不完整：期望 {remote_size} 字节，实际 {actual} 字节"
        )
    print(f"[done] {dest.name} {actual/1e6:.0f} MB")
    return dest
