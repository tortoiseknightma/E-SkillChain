"""Create a bounded Recipe1M+ inventory from human-reviewed title aliases."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from skillchain.data.recipe1m_plus_selection import (  # noqa: E402
    RecipeSelectionError,
    build_selection,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers-archive", type=Path, required=True)
    parser.add_argument("--det-ingrs", type=Path, required=True)
    parser.add_argument("--layer2-plus", type=Path, required=True)
    parser.add_argument("--reviewed-map", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--aria2-input", type=Path, required=True)
    args = parser.parse_args()
    try:
        inventory = build_selection(
            layers_archive=args.layers_archive,
            det_ingrs=args.det_ingrs,
            layer2_plus=args.layer2_plus,
            reviewed_mapping=args.reviewed_map,
            inventory_path=args.inventory,
            aria2_input_path=args.aria2_input,
        )
    except RecipeSelectionError as error:
        parser.error(str(error))
    print(
        f"recipes={inventory['recipe_count']} images={inventory['image_count']} "
        f"inventory={args.inventory}"
    )


if __name__ == "__main__":
    main()
