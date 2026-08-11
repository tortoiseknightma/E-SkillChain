"""从 MUGE 商品固定分配 Exact Match 与 Divergent Recommendation 查询池。

服饰标题进入 Divergent Rec.，其余商品进入 Exact Match；分配只依赖标题，和两个
命令的执行顺序无关。DeepFashion 仍可作为后续质量补充。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq

from skillchain import config
from skillchain.data import (
    discard_staging_directory,
    prepare_staging_directory,
    publish_staged_directory,
)

PRODUCTS = config.DATA_DIR / "clean" / "products.parquet"
DESTINATION = config.DATA_DIR / "clean" / "query_images" / "divergent_rec"

FASHION_KEYWORDS = (
    "连衣裙", "半身裙", "包臀裙", "短裙", "长裙", "裤", "T恤", "t恤",
    "衬衫", "上衣", "卫衣", "毛衣", "针织衫", "外套", "夹克", "西装",
    "风衣", "睡衣", "内衣", "文胸", "袜", "运动鞋", "凉鞋", "皮鞋",
    "板鞋", "拖鞋", "靴", "帽", "围巾", "腰带", "女包", "男包",
    "手提包", "斜挎包", "双肩包", "腰包", "背包", "包包",
)


def is_fashion_title(title: str) -> bool:
    return any(keyword in title for keyword in FASHION_KEYWORDS)


def select_fashion_products(
    rows: Iterable[dict], limit: int = 800, exclude_product_ids: set[str] | None = None
) -> list[dict]:
    exclude_product_ids = exclude_product_ids or set()
    selected = []
    for row in rows:
        if row.get("product_id") not in exclude_product_ids and is_fashion_title(
            str(row.get("title", ""))
        ):
            selected.append(row)
            if len(selected) >= limit:
                break
    return selected


def select_exact_match_products(rows: Iterable[dict], limit: int = 2_000) -> list[dict]:
    """固定选择非服饰商品，保证与 Fashion 池双向互斥。"""
    selected = []
    for row in rows:
        if not is_fashion_title(str(row.get("title", ""))):
            selected.append(row)
            if len(selected) >= limit:
                break
    return selected


def materialize(products_path: Path, destination: Path, limit: int = 800) -> int:
    products_path = Path(products_path)
    destination = Path(destination)
    rows = pq.read_table(products_path).to_pylist()
    selected = select_fashion_products(rows, limit=limit)
    staging = prepare_staging_directory(destination)
    try:
        with (staging / "manifest.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as manifest:
            for row in selected:
                source = products_path.parent / row["image_path"]
                filename = f"{row['product_id']}.jpg"
                target = staging / filename
                try:
                    target.hardlink_to(source)
                except OSError:
                    shutil.copy2(source, target)
                manifest.write(
                    json.dumps(
                        {
                            "image": filename,
                            "product_id": row["product_id"],
                            "title": row["title"],
                            "source": row["source"],
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        if len(list(staging.glob("muge-*.jpg"))) != len(selected):
            raise RuntimeError("服饰查询图 staging 数量校验失败")
        publish_staged_directory(staging, destination)
    except Exception:
        discard_staging_directory(staging)
        raise
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=800)
    args = parser.parse_args()
    count = materialize(PRODUCTS, DESTINATION, limit=args.limit)
    print(f"query_images={count}")
    print(f"manifest={DESTINATION / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
