"""Replace unavailable Recipe1M+ recipes within the reviewed entity bounds."""

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
    repair_missing_recipes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--layers-archive", type=Path, required=True)
    parser.add_argument("--det-ingrs", type=Path, required=True)
    parser.add_argument("--layer2-plus", type=Path, required=True)
    parser.add_argument("--reviewed-map", type=Path, required=True)
    parser.add_argument("--repaired-inventory", type=Path, required=True)
    parser.add_argument("--aria2-input", type=Path, required=True)
    parser.add_argument("--prior-inventory", type=Path)
    args = parser.parse_args()
    try:
        inventory = repair_missing_recipes(
            inventory_path=args.inventory,
            layers_archive=args.layers_archive,
            det_ingrs=args.det_ingrs,
            layer2_plus=args.layer2_plus,
            reviewed_mapping=args.reviewed_map,
            repaired_inventory_path=args.repaired_inventory,
            aria2_input_path=args.aria2_input,
            prior_inventory_path=args.prior_inventory,
        )
    except RecipeSelectionError as error:
        parser.error(str(error))
    print(
        f"recipes={inventory['recipe_count']} images={inventory['image_count']} "
        f"replaced={inventory['replaced_recipe_count']}"
    )


if __name__ == "__main__":
    main()
