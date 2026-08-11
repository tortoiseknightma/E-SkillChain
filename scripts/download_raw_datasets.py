"""Run the cross-session raw dataset downloader from a source checkout."""

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skillchain.data.raw_download import main  # noqa: E402


if __name__ == "__main__":
    main()
