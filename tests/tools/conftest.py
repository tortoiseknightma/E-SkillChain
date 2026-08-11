from collections.abc import Callable
from typing import Any

import pytest

from skillchain.schemas import Product

ProductFactory = Callable[..., Product]


@pytest.fixture
def product_factory() -> ProductFactory:
    def make_product(**overrides: Any) -> Product:
        values = {
            "product_id": "product-1",
            "title": "蓝色连衣裙",
            "category_l1": "女装",
            "category_l2": "连衣裙",
            "image_path": "images/product-1.jpg",
            "source": "muge",
        }
        values.update(overrides)
        return Product.model_validate(values)

    return make_product


@pytest.fixture
def product(product_factory: ProductFactory) -> Product:
    return product_factory()
