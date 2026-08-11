import pytest
from pydantic import ValidationError

from skillchain.schemas import KBEntry, Product


def test_product_accepts_optional_mep3m_category_and_ocr_fields():
    product = Product(
        product_id="mep3m-34-1",
        title="red dress",
        category_l1="clothing",
        category_l2="dress",
        category_l3="evening dress",
        ocr_text="SALE",
        image_path="product_images/mep3m/mep3m-34-1.jpg",
        source="mep3m",
    )

    assert product.category_l3 == "evening dress"
    assert product.ocr_text == "SALE"


def test_product_defaults_optional_mep3m_fields_to_none():
    product = Product(
        product_id="muge-1",
        title="red dress",
        category_l1="unknown",
        image_path="product_images/muge/muge-1.jpg",
        source="muge",
    )

    assert product.category_l3 is None
    assert product.ocr_text is None


def test_product_accepts_current_formal_portfolio_sources():
    for source in ("abo", "rpc", "fashioniq"):
        product = Product(
            product_id=f"{source}-1",
            title="商品",
            category_l1="test",
            image_path=f"{source}/1.jpg",
            source=source,
        )
        assert product.source == source


def test_product_rejects_blank_source():
    with pytest.raises(ValidationError):
        Product(
            product_id="product-1",
            title="商品",
            category_l1="test",
            image_path="product.jpg",
            source=" ",
        )


def test_llm_synth_kb_entry_records_actual_provider_and_model():
    entry = KBEntry(
        entry_id="synth-1",
        title="长尾条目",
        text="合成内容",
        kind="encyclopedia",
        origin="llm_synth",
        synth_provider="gpt_5_6_sol",
        synth_model="GPT 5.6 Sol",
    )

    assert entry.model_dump()["synth_provider"] == "gpt_5_6_sol"
    assert entry.model_dump()["synth_model"] == "GPT 5.6 Sol"


def test_llm_synth_kb_entry_rejects_missing_provenance():
    with pytest.raises(ValidationError, match="synth_provider"):
        KBEntry(
            entry_id="synth-1",
            title="长尾条目",
            text="合成内容",
            kind="encyclopedia",
            origin="llm_synth",
        )


def test_llm_synth_kb_entry_rejects_blank_model_name():
    with pytest.raises(ValidationError, match="synth_model"):
        KBEntry(
            entry_id="synth-1",
            title="长尾条目",
            text="合成内容",
            kind="encyclopedia",
            origin="llm_synth",
            synth_provider="fable",
            synth_model="   ",
        )
