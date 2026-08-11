"""Deterministically build and verify the frozen authoring packet variants."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from skillchain import config
from skillchain.static_authoring import (
    AuthoringBudgets,
    FixedDecoding,
    ModelIdentity,
    PriceSchedule,
    PromptIdentity,
    build_authoring_packet,
    load_verified_price_schedule,
)
from skillchain.task_spec import load_default_task_specification
from skillchain.taxonomy import load_default_taxonomy_registry
from skillchain.tools.registry import build_mvp_registry_spec
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[1]
AUTHORING_ROOT = ROOT / "specs" / "authoring"
PROMPT_PATH = AUTHORING_ROOT / "static-author-v3.txt"
PRICING_SOURCE_URL = "https://help.aliyun.com/zh/model-studio/model-pricing"
FROZEN_ON = "2026-07-23"
CONVERSION_MICROUSD_PER_CNY = 150_000


VARIANTS = {
    "fallback": {
        "model": config.LEGACY_AUTHOR_FALLBACK_MODEL,
        "revision": config.LEGACY_AUTHOR_FALLBACK_MODEL_REVISION,
        "source_input": 1_000_000,
        "source_output": 10_000_000,
    },
    "primary": {
        "model": config.LEGACY_AUTHOR_MODEL,
        "revision": config.LEGACY_AUTHOR_MODEL_REVISION,
        "source_input": 150_000,
        "source_output": 1_500_000,
    },
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _converted(source_microunits: int) -> int:
    return (source_microunits * CONVERSION_MICROUSD_PER_CNY + 999_999) // 1_000_000


def _schedule(name: str, settings: dict[str, object]) -> PriceSchedule:
    model = str(settings["model"])
    source_input = int(settings["source_input"])
    source_output = int(settings["source_output"])
    return PriceSchedule(
        schedule_id=f"dashscope-beijing-{model}-2026-07-22-v1",
        provider="qwen",
        model=model,
        input_microusd_per_million_tokens=_converted(source_input),
        output_microusd_per_million_tokens=_converted(source_output),
        source_input_microunits_per_million_tokens=source_input,
        source_output_microunits_per_million_tokens=source_output,
        conversion_microusd_per_source_unit=CONVERSION_MICROUSD_PER_CNY,
        tier_max_input_tokens=32_000,
        source_url=PRICING_SOURCE_URL,
        source_revision="retrieved-2026-07-22",
    )


def _expected_files() -> dict[Path, bytes]:
    prompt_bytes = PROMPT_PATH.read_bytes()
    if not prompt_bytes.endswith(b"\n") or prompt_bytes.endswith(b"\n\n"):
        raise ValueError("prompt source must end in exactly one LF")
    prompt_text = prompt_bytes[:-1].decode("utf-8", errors="strict")
    prompt = PromptIdentity(
        prompt_id="static-author-v3",
        prompt_version="3.0.0",
        template=prompt_text,
        prompt_sha256=sha256_bytes(prompt_text.encode("utf-8")),
    )
    taxonomy = load_default_taxonomy_registry()
    tasks = load_default_task_specification()
    registry = build_mvp_registry_spec()
    decoding = FixedDecoding(seed=20260722, max_output_tokens=8_000)
    budgets = AuthoringBudgets(
        max_input_tokens=32_000,
        max_output_tokens=8_000,
        max_total_tokens=40_000,
        max_cost_microusd=50_000,
        max_human_review_minutes=30,
    )
    generated: dict[Path, bytes] = {}
    variant_locks: dict[str, object] = {}
    for name, settings in sorted(VARIANTS.items()):
        schedule = _schedule(name, settings)
        schedule_path = AUTHORING_ROOT / f"price-{settings['model']}-v1.json"
        schedule_bytes = canonical_json_bytes(schedule.model_dump(mode="json"))
        generated[schedule_path] = schedule_bytes
        verified_schedule = load_verified_price_schedule_from_bytes(
            schedule_path, schedule_bytes, generated
        )
        packet = build_authoring_packet(
            taxonomy=taxonomy,
            task_specification=tasks,
            tool_registry=registry,
            prompt=prompt,
            model=ModelIdentity(
                provider="qwen",
                endpoint=config.PROVIDER_ENDPOINTS["qwen"],
                model=str(settings["model"]),
                revision=str(settings["revision"]),
            ),
            decoding=decoding,
            price_schedule=verified_schedule,
            budgets=budgets,
            public_sources=(),
            reference_skill_bundle=None,
            defer_tool_registry_runtime=True,
        )
        packet_path = AUTHORING_ROOT / f"authoring-packet-{name}-v3-r2.json"
        packet_bytes = packet.canonical_bytes()
        generated[packet_path] = packet_bytes
        worst_cost = (
            budgets.max_input_tokens * schedule.input_microusd_per_million_tokens
            + budgets.max_output_tokens * schedule.output_microusd_per_million_tokens
            + 999_999
        ) // 1_000_000
        variant_locks[name] = {
            "input_sha256": packet.input_sha256,
            "model": settings["model"],
            "model_revision": settings["revision"],
            "packet_file": packet_path.relative_to(ROOT).as_posix(),
            "packet_file_sha256": sha256_bytes(packet_bytes),
            "price_schedule_file": schedule_path.relative_to(ROOT).as_posix(),
            "price_schedule_file_sha256": sha256_bytes(schedule_bytes),
            "worst_case_cost_microusd": worst_cost,
        }
    access = {
        "schema_version": 1,
        "checked_on": FROZEN_ON,
        "endpoint": config.PROVIDER_ENDPOINTS["qwen"],
        "method": "read-only models.list; no generation request",
        "provider": "qwen",
        "required_snapshot_models": sorted(
            str(item["model"]) for item in VARIANTS.values()
        ),
        "snapshots_observed": [
            "qwen3-vl-flash-2025-10-15",
            "qwen3-vl-flash-2026-01-22",
            "qwen3-vl-plus-2025-09-23",
            "qwen3-vl-plus-2025-12-19",
        ],
    }
    access_path = AUTHORING_ROOT / "model-access-evidence-v2.json"
    access_bytes = canonical_json_bytes(access)
    generated[access_path] = access_bytes
    sandbox_manifest = ROOT / "deploy" / "authoring" / "locks" / "lock-manifest.json"
    freeze = {
        "schema_version": 1,
        "status": "frozen",
        "freeze_id": "authoring-freeze-2026-07-23-v3",
        "approved_on": FROZEN_ON,
        "approved_by": "project-owner",
        "default_variant": "primary",
        "fallback_policy": "manual-selection-before-run-only; no automatic fallback",
        "variants": variant_locks,
        "common": {
            "budgets": budgets.model_dump(mode="json"),
            "decoding": decoding.model_dump(mode="json"),
            "exchange_rate_policy": {
                "conversion_microusd_per_cny": CONVERSION_MICROUSD_PER_CNY,
                "meaning": "conservative budget conversion; 1 CNY = 0.15 USD",
            },
            "model_access_evidence_file": access_path.relative_to(ROOT).as_posix(),
            "model_access_evidence_file_sha256": sha256_bytes(access_bytes),
            "pricing_source_url": PRICING_SOURCE_URL,
            "prompt_file": PROMPT_PATH.relative_to(ROOT).as_posix(),
            "prompt_file_sha256": _sha(PROMPT_PATH),
            "prompt_text_sha256": prompt.prompt_sha256,
            "prompt_transform": "utf8-strip-one-final-lf-v1",
            "public_source_ids": [],
            "reference_skill_bundle": None,
            "sandbox_lock_manifest_file": sandbox_manifest.relative_to(ROOT).as_posix(),
            "sandbox_lock_manifest_file_sha256": _sha(sandbox_manifest),
            "task_specification_sha256": tasks.task_spec_sha256,
            "taxonomy_sha256": taxonomy.taxonomy_sha256,
            "tool_registry_runtime_policy": "deferred_until_bank_compile",
            "tool_registry_sha256": registry.registry_sha256,
        },
    }
    freeze_path = AUTHORING_ROOT / "authoring-freeze-lock-v3.json"
    generated[freeze_path] = canonical_json_bytes(freeze)
    return generated


def load_verified_price_schedule_from_bytes(
    path: Path, content: bytes, generated: dict[Path, bytes]
):
    """Load through the production verifier without leaving unchecked bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_bytes() if path.exists() else None
    path.write_bytes(content)
    try:
        return load_verified_price_schedule(
            path, expected_file_sha256=sha256_bytes(content)
        )
    finally:
        if existing is None:
            path.unlink()
        else:
            path.write_bytes(existing)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = _expected_files()
    if args.check:
        mismatches = [
            path.relative_to(ROOT).as_posix()
            for path, content in expected.items()
            if not path.is_file() or path.read_bytes() != content
        ]
        if mismatches:
            raise SystemExit("authoring freeze drift: " + ", ".join(mismatches))
        return 0
    for path, content in expected.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
