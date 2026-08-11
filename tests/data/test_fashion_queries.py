import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from skillchain.data.fashion_queries import (
    materialize,
    select_exact_match_products,
    select_fashion_products,
)


def test_select_fashion_products_uses_precise_keywords_and_stable_limit():
    rows = [
        {"product_id": "1", "title": "全自动洗衣机", "image_path": "1.jpg"},
        {"product_id": "2", "title": "女士连衣裙", "image_path": "2.jpg"},
        {"product_id": "3", "title": "复古斜挎包", "image_path": "3.jpg"},
        {"product_id": "4", "title": "男士运动鞋", "image_path": "4.jpg"},
    ]

    selected = select_fashion_products(rows, limit=2)

    assert [row["product_id"] for row in selected] == ["2", "3"]


def test_select_fashion_products_excludes_exact_match_pool():
    rows = [
        {"product_id": "1", "title": "女士连衣裙", "image_path": "1.jpg"},
        {"product_id": "2", "title": "男士运动鞋", "image_path": "2.jpg"},
    ]

    selected = select_fashion_products(rows, limit=1, exclude_product_ids={"1"})

    assert [row["product_id"] for row in selected] == ["2"]


def test_exact_and_fashion_partition_is_disjoint_and_order_independent():
    rows = [
        {"product_id": "1", "title": "女士连衣裙", "image_path": "1.jpg"},
        {"product_id": "2", "title": "手机壳", "image_path": "2.jpg"},
        {"product_id": "3", "title": "男士运动鞋", "image_path": "3.jpg"},
        {"product_id": "4", "title": "蓝牙耳机", "image_path": "4.jpg"},
    ]

    fashion_first = select_fashion_products(rows, limit=10)
    exact_second = select_exact_match_products(rows, limit=10)
    exact_first = select_exact_match_products(rows, limit=10)
    fashion_second = select_fashion_products(rows, limit=10)

    assert {row["product_id"] for row in fashion_first} == {"1", "3"}
    assert {row["product_id"] for row in exact_second} == {"2", "4"}
    assert exact_first == exact_second
    assert fashion_first == fashion_second


def test_materialize_hardlinks_images_and_writes_manifest(tmp_path):
    clean_dir = tmp_path / "clean"
    images_dir = clean_dir / "product_images"
    images_dir.mkdir(parents=True)
    for name in ("dress.jpg", "phone.jpg"):
        Image.new("RGB", (240, 240), "red").save(images_dir / name)
    rows = [
        {
            "product_id": "muge-1",
            "title": "女士连衣裙",
            "category_l1": "unknown",
            "category_l2": None,
            "image_path": "product_images/dress.jpg",
            "source": "muge",
        },
        {
            "product_id": "muge-2",
            "title": "手机壳",
            "category_l1": "unknown",
            "category_l2": None,
            "image_path": "product_images/phone.jpg",
            "source": "muge",
        },
    ]
    parquet_path = clean_dir / "products.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)

    destination = clean_dir / "query_images" / "divergent_rec"
    count = materialize(parquet_path, destination, limit=10)

    assert count == 1
    assert (destination / "muge-1.jpg").is_file()
    manifest = [
        json.loads(line)
        for line in (destination / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manifest == [
        {
            "image": "muge-1.jpg",
            "product_id": "muge-1",
            "title": "女士连衣裙",
            "source": "muge",
        }
    ]


def test_materialize_failure_preserves_previous_manifest(tmp_path):
    clean_dir = tmp_path / "clean"
    images_dir = clean_dir / "product_images"
    images_dir.mkdir(parents=True)
    Image.new("RGB", (240, 240), "red").save(images_dir / "dress.jpg")
    rows = [
        {"product_id": "muge-1", "title": "女士连衣裙", "category_l1": "unknown", "category_l2": None, "image_path": "product_images/dress.jpg", "source": "muge"},
        {"product_id": "muge-2", "title": "男士运动鞋", "category_l1": "unknown", "category_l2": None, "image_path": "product_images/missing.jpg", "source": "muge"},
    ]
    parquet_path = clean_dir / "products.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    destination = clean_dir / "query_images" / "divergent_rec"
    destination.mkdir(parents=True)
    manifest = destination / "manifest.jsonl"
    manifest.write_text('{"old":true}\n', encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        materialize(parquet_path, destination, limit=2)

    assert manifest.read_text(encoding="utf-8") == '{"old":true}\n'
    assert not (destination / "muge-1.jpg").exists()
