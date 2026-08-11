"""Phase 1 数据获取与清洗工具。"""

import os
import shutil
from pathlib import Path


def normalize_line_separators(value: str) -> str:
    """把会被部分 JSONL 消费者视为记录边界的 Unicode 分隔符归一为换行。"""
    return value.replace("\u2028", "\n").replace("\u2029", "\n")


def prepare_staging_directory(destination: Path) -> Path:
    """创建同盘 staging 目录，并恢复上次交换中断留下的 backup。"""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.staging")
    backup = destination.with_name(f".{destination.name}.backup")
    if backup.exists() and not destination.exists():
        os.replace(backup, destination)
    elif backup.exists():
        shutil.rmtree(backup)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    return staging


def discard_staging_directory(staging: Path) -> None:
    if staging.exists():
        shutil.rmtree(staging)


def publish_staged_directory(staging: Path, destination: Path) -> None:
    """用同盘目录重命名提交完整集合；交换失败时恢复旧目录。"""
    destination = Path(destination)
    backup = destination.with_name(f".{destination.name}.backup")
    if backup.exists():
        shutil.rmtree(backup)
    had_previous = destination.exists()
    if had_previous:
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except Exception:
        if had_previous and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup.exists():
        try:
            shutil.rmtree(backup)
        except OSError:
            # The new destination is already committed.  A later staging run
            # will clean this stale backup without reporting a false failure.
            pass


def publish_staged_directory_and_file(
    staging_directory: Path,
    destination_directory: Path,
    staging_file: Path,
    destination_file: Path,
) -> None:
    """联合发布一个目录和一个文件；任一步失败都恢复旧代产物。"""
    destination_directory = Path(destination_directory)
    destination_file = Path(destination_file)
    directory_backup = destination_directory.with_name(
        f".{destination_directory.name}.backup"
    )
    file_backup = destination_file.with_name(f".{destination_file.name}.backup")
    recover_directory_and_file_publish(destination_directory, destination_file)
    moved_old_directory = moved_old_file = False
    published_directory = published_file = False
    try:
        if destination_directory.exists():
            os.replace(destination_directory, directory_backup)
            moved_old_directory = True
        if destination_file.exists():
            os.replace(destination_file, file_backup)
            moved_old_file = True
        os.replace(staging_directory, destination_directory)
        published_directory = True
        os.replace(staging_file, destination_file)
        published_file = True
    except Exception:
        if published_file:
            destination_file.unlink(missing_ok=True)
        if published_directory and destination_directory.exists():
            shutil.rmtree(destination_directory)
        if moved_old_file and file_backup.exists():
            os.replace(file_backup, destination_file)
        if moved_old_directory and directory_backup.exists():
            os.replace(directory_backup, destination_directory)
        raise
    if directory_backup.exists():
        shutil.rmtree(directory_backup)
    file_backup.unlink(missing_ok=True)


def recover_directory_and_file_publish(
    destination_directory: Path, destination_file: Path
) -> None:
    """Recover or finish an interrupted directory-and-file atomic publish."""
    destination_directory = Path(destination_directory)
    destination_file = Path(destination_file)
    directory_backup = destination_directory.with_name(
        f".{destination_directory.name}.backup"
    )
    file_backup = destination_file.with_name(f".{destination_file.name}.backup")
    has_directory_backup = directory_backup.exists()
    has_file_backup = file_backup.exists()
    if not has_directory_backup and not has_file_backup:
        return

    if has_directory_backup and has_file_backup:
        if destination_directory.exists() and destination_file.exists():
            shutil.rmtree(directory_backup)
            file_backup.unlink()
            return
        if destination_directory.exists():
            shutil.rmtree(destination_directory)
        destination_file.unlink(missing_ok=True)
        os.replace(directory_backup, destination_directory)
        os.replace(file_backup, destination_file)
        return

    if has_directory_backup:
        if destination_directory.exists():
            shutil.rmtree(directory_backup)
        else:
            os.replace(directory_backup, destination_directory)
        return

    if destination_file.exists():
        file_backup.unlink()
    else:
        if destination_directory.exists():
            shutil.rmtree(destination_directory)
        os.replace(file_backup, destination_file)
