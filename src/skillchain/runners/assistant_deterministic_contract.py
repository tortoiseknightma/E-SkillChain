"""Versioned deterministic action/response contract for Core Fast.

The language model still owns capability routing.  Once a routed capability is
selected, this module owns the required tool sequence and compiles the public
tool DTO into the exact user-visible response surface.  It deliberately uses
no scorer sidecar, query label, private product identity, or model-generated
facts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import re
from typing import Literal

from skillchain.tools.contracts import JSONValue, validate_json_value
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes


DETERMINISTIC_ASSISTANT_CONTRACT_VERSION = "core-fast-deterministic-action-response-v4"
DETERMINISTIC_SEMANTIC_POLICY_VERSION = "core-fast-semantic-policy-v2"

SEMANTIC_POLICY_BEGIN = "<!-- skillchain-semantic-policy-v2"
SEMANTIC_POLICY_END = "-->"

CapabilityId = Literal[
    "knowledge.visual_encyclopedia",
    "product.exact_match",
    "product.multi_search",
    "product.style_recommendation",
    "utility.document_reading",
    "utility.recipe_guidance",
]

SEMANTIC_POLICY_CAPABILITIES: tuple[CapabilityId, ...] = (
    "knowledge.visual_encyclopedia",
    "product.style_recommendation",
    "utility.document_reading",
    "utility.recipe_guidance",
)


@dataclass(frozen=True)
class DeterministicToolObservation:
    tool_name: str
    status: Literal["success", "error"]
    public_output: Mapping[str, object] | None = None


@dataclass(frozen=True)
class DeterministicToolDecision:
    tool_name: str
    arguments: dict[str, JSONValue]


@dataclass(frozen=True)
class DeterministicSemanticPolicy:
    """The only S1-authored surface consumed by the deterministic compiler.

    The policy is deliberately narrow: the Creator may select literal public
    evidence atoms and choose whether to abstain when none remain.  Tool order,
    tool arguments, output sections, handles, cards, and fallback markers stay
    runner/compiler owned.
    """

    capability_id: CapabilityId
    evidence_terms: tuple[str, ...] = ()
    require_all_terms: bool = False
    abstain_when_no_evidence: bool = True
    ocr_extraction_plan: Literal["all-lines", "literal-material-spans"] = "all-lines"


_SEMANTIC_TOKEN_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,80}$")
_MATERIAL_SEGMENT_RE = re.compile(r"(?<=[。！？.!?])\s+|\n+")
_MATERIAL_TEXT_RE = re.compile(r"[A-Za-z0-9\u3400-\u9fff]")


def _semantic_policy_payload(
    policy: DeterministicSemanticPolicy,
) -> dict[str, JSONValue]:
    if policy.capability_id not in CapabilityId.__args__:
        raise ValueError("semantic policy capability is unknown")
    normalized = tuple(term.strip() for term in policy.evidence_terms)
    if (
        normalized != policy.evidence_terms
        or normalized != tuple(sorted(set(normalized), key=str.casefold))
        or any(
            not term or _SEMANTIC_TOKEN_RE.fullmatch(term) is None
            for term in normalized
        )
    ):
        raise ValueError("semantic policy evidence terms must be canonical")
    if (
        policy.ocr_extraction_plan != "all-lines"
        and policy.capability_id != "utility.document_reading"
    ):
        raise ValueError("OCR extraction plans are Document-only")
    return {
        "policy_version": DETERMINISTIC_SEMANTIC_POLICY_VERSION,
        "capability_id": policy.capability_id,
        "evidence_terms": list(policy.evidence_terms),
        "require_all_terms": policy.require_all_terms,
        "abstain_when_no_evidence": policy.abstain_when_no_evidence,
        "ocr_extraction_plan": policy.ocr_extraction_plan,
    }


def render_deterministic_semantic_policy(
    policy: DeterministicSemanticPolicy,
) -> str:
    """Render a canonical compiler-consumed Bank-body extension."""

    payload = _semantic_policy_payload(policy)
    return (
        f"{SEMANTIC_POLICY_BEGIN}\n"
        + canonical_json_bytes(payload).decode("utf-8")
        + f"\n{SEMANTIC_POLICY_END}\n"
    )


def parse_deterministic_semantic_policy(
    body: str,
    *,
    capability_id: str,
) -> DeterministicSemanticPolicy:
    """Read one optional typed policy from canonical Skill Body bytes.

    Absence is the byte-compatible parent default.  Malformed or capability-
    mismatched policy text fails closed instead of silently becoming prose.
    """

    start = body.find(SEMANTIC_POLICY_BEGIN)
    if start < 0:
        return DeterministicSemanticPolicy(capability_id=capability_id)  # type: ignore[arg-type]
    if body.find(SEMANTIC_POLICY_BEGIN, start + len(SEMANTIC_POLICY_BEGIN)) >= 0:
        raise ValueError("Skill Body contains multiple semantic policies")
    payload_start = start + len(SEMANTIC_POLICY_BEGIN)
    if not body.startswith("\n", payload_start):
        raise ValueError("semantic policy header is not canonical")
    end = body.find(f"\n{SEMANTIC_POLICY_END}\n", payload_start + 1)
    if end < 0:
        raise ValueError("semantic policy footer is missing")
    raw = body[payload_start + 1 : end]
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as error:
        raise ValueError("semantic policy payload is invalid JSON") from error
    if not isinstance(payload, dict) or set(payload) != {
        "policy_version",
        "capability_id",
        "evidence_terms",
        "require_all_terms",
        "abstain_when_no_evidence",
        "ocr_extraction_plan",
    }:
        raise ValueError("semantic policy payload shape is invalid")
    terms = payload["evidence_terms"]
    if not isinstance(terms, list) or not all(isinstance(item, str) for item in terms):
        raise ValueError("semantic policy evidence terms are invalid")
    if (
        payload["policy_version"] != DETERMINISTIC_SEMANTIC_POLICY_VERSION
        or payload["capability_id"] != capability_id
        or not isinstance(payload["require_all_terms"], bool)
        or not isinstance(payload["abstain_when_no_evidence"], bool)
        or payload["ocr_extraction_plan"] not in {"all-lines", "literal-material-spans"}
    ):
        raise ValueError("semantic policy identity is invalid")
    policy = DeterministicSemanticPolicy(
        capability_id=capability_id,  # type: ignore[arg-type]
        evidence_terms=tuple(terms),
        require_all_terms=payload["require_all_terms"],
        abstain_when_no_evidence=payload["abstain_when_no_evidence"],
        ocr_extraction_plan=payload["ocr_extraction_plan"],
    )
    if canonical_json_bytes(_semantic_policy_payload(policy)).decode("utf-8") != raw:
        raise ValueError("semantic policy payload is not canonical")
    return policy


def deterministic_contract_payload() -> dict[str, JSONValue]:
    payload: dict[str, JSONValue] = {
        "policy_version": DETERMINISTIC_ASSISTANT_CONTRACT_VERSION,
        "scope": "core_fast_routed_configs_only",
        "route_owner": "model_selected_capability_from_frozen_descriptions",
        "tool_owner": "runner",
        "response_owner": "deterministic_public_dto_compiler",
        "semantic_policy_owner": "s1_typed_bank_surface",
        "semantic_policy_version": DETERMINISTIC_SEMANTIC_POLICY_VERSION,
        "private_or_scorer_inputs": "forbidden",
        "contracts": {
            "knowledge.visual_encyclopedia": {
                "tools": ["object_detect", "encyclopedia_lookup"],
                "fallback_marker": "not enough evidence",
            },
            "product.exact_match": {
                "tools": ["image_product_search"],
                "fallback_marker": "no supported match",
            },
            "product.multi_search": {
                "tools": ["multi_product_search"],
                "fallback_marker": "no supported item",
            },
            "product.style_recommendation": {
                "tools": ["style_similar_search"],
                "fallback_marker": "unable to recommend",
            },
            "utility.document_reading": {
                "tools": ["document_ocr"],
                "fallback_marker": "unable to read",
            },
            "utility.recipe_guidance": {
                "tools": ["object_detect", "recipe_lookup"],
                "fallback_marker": "no supported recipe",
            },
        },
    }
    validate_json_value(payload)
    return payload


DETERMINISTIC_ASSISTANT_CONTRACT_SHA256 = sha256_bytes(
    canonical_json_bytes(deterministic_contract_payload())
)


def _successful(
    observations: Sequence[DeterministicToolObservation], tool_name: str
) -> DeterministicToolObservation | None:
    return next(
        (
            item
            for item in reversed(observations)
            if item.tool_name == tool_name and item.status == "success"
        ),
        None,
    )


def _attempted(
    observations: Sequence[DeterministicToolObservation], tool_name: str
) -> bool:
    return any(item.tool_name == tool_name for item in observations)


def _first_detection_label(
    observation: DeterministicToolObservation | None,
) -> str | None:
    if observation is None or not isinstance(observation.public_output, Mapping):
        return None
    detections = observation.public_output.get("detections")
    if not isinstance(detections, Sequence) or isinstance(detections, (str, bytes)):
        return None
    for item in detections:
        if not isinstance(item, Mapping):
            continue
        label = item.get("label")
        if isinstance(label, str) and label.strip():
            return label.strip()
    return None


def next_deterministic_tool(
    capability_id: str,
    observations: Sequence[DeterministicToolObservation],
) -> DeterministicToolDecision | None:
    """Return the next runner-owned tool call, or ``None`` when compilable."""

    single_tools = {
        "product.exact_match": "image_product_search",
        "product.multi_search": "multi_product_search",
        "product.style_recommendation": "style_similar_search",
        "utility.document_reading": "document_ocr",
    }
    tool_name = single_tools.get(capability_id)
    if tool_name is not None:
        arguments: dict[str, JSONValue] = (
            {} if tool_name == "text_product_search" else {"asset_id": "query_asset"}
        )
        return (
            None
            if _attempted(observations, tool_name)
            else DeterministicToolDecision(
                tool_name=tool_name,
                arguments=arguments,
            )
        )

    lookup = {
        "knowledge.visual_encyclopedia": ("encyclopedia_lookup", "entity"),
        "utility.recipe_guidance": ("recipe_lookup", "dish"),
    }.get(capability_id)
    if lookup is None:
        return None
    if not _attempted(observations, "object_detect"):
        return DeterministicToolDecision(
            tool_name="object_detect",
            arguments={"asset_id": "query_asset"},
        )
    detected = _first_detection_label(_successful(observations, "object_detect"))
    lookup_tool, argument_name = lookup
    if detected is None or _attempted(observations, lookup_tool):
        return None
    return DeterministicToolDecision(
        tool_name=lookup_tool,
        arguments={argument_name: detected},
    )


def deterministic_tool_names(capability_id: str) -> tuple[str, ...]:
    contracts = deterministic_contract_payload()["contracts"]
    if not isinstance(contracts, Mapping):  # pragma: no cover - frozen payload
        raise AssertionError("deterministic contracts must be an object")
    contract = contracts.get(capability_id)
    if not isinstance(contract, Mapping):
        return ()
    tools = contract.get("tools")
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)):
        return ()
    return tuple(item for item in tools if isinstance(item, str))


def _output(
    observations: Sequence[DeterministicToolObservation], tool_name: str
) -> Mapping[str, object] | None:
    item = _successful(observations, tool_name)
    return None if item is None else item.public_output


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _sequence(value: object) -> Sequence[object]:
    return (
        value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        else ()
    )


def _policy_matches(text: str, policy: DeterministicSemanticPolicy) -> bool:
    if not policy.evidence_terms:
        return True
    folded = text.casefold()
    checks = tuple(term.casefold() in folded for term in policy.evidence_terms)
    return all(checks) if policy.require_all_terms else any(checks)


def _candidate_card(candidate: Mapping[str, object]) -> str | None:
    title = _text(candidate.get("title"))
    evidence = _text(candidate.get("evidence_reference"))
    product = _text(candidate.get("product_id"))
    if None in (title, evidence, product):
        return None
    return f"{evidence} | {product} | {title}"


def _compile_product(output: Mapping[str, object] | None) -> str:
    candidates = _sequence(None if output is None else output.get("candidates"))
    cards = [
        rendered
        for item in candidates
        if isinstance(item, Mapping)
        if (rendered := _candidate_card(item)) is not None
    ]
    if not cards:
        return (
            "answer:\nno supported match\n"
            "product_cards:\nnone\n"
            "uncertainty:\nNo eligible candidate was returned by the tool."
        )
    return (
        "answer:\nThe tool returned eligible candidates listed below.\n"
        f"product_cards:\n{'\n'.join(cards)}\n"
        "uncertainty:\nOnly the returned public candidate evidence is shown."
    )


def _compile_multi(output: Mapping[str, object] | None) -> str:
    raw_items = _sequence(None if output is None else output.get("items"))
    mapping_lines: list[str] = []
    cards_by_ordinal: dict[int, str] = {}
    matched = False
    for raw in raw_items:
        if not isinstance(raw, Mapping):
            continue
        item_ref = _text(raw.get("item_ref"))
        label = _text(raw.get("label")) or "detected item"
        status = raw.get("status")
        ordinal = raw.get("candidate_ordinal")
        if item_ref is None or status not in {"matched", "unresolved"}:
            continue
        line = f"{item_ref} | {label} | {status}"
        if status == "matched" and isinstance(ordinal, int) and ordinal > 0:
            line += f" | candidate-{ordinal}"
            candidate = raw.get("candidate")
            if isinstance(candidate, Mapping):
                rendered = _candidate_card(candidate)
                if rendered is not None:
                    cards_by_ordinal.setdefault(ordinal, rendered)
                    matched = True
        mapping_lines.append(line)
    mapping = "\n".join(mapping_lines) or "none"
    if not matched:
        return (
            "answer:\nno supported item\n"
            f"item_mapping:\n{mapping}\n"
            "product_cards:\nnone\n"
            "uncertainty:\nNo matched public candidate was returned by the tool."
        )
    cards = "\n".join(cards_by_ordinal[key] for key in sorted(cards_by_ordinal))
    return (
        "answer:\nThe item mapping and matched candidates are listed below.\n"
        f"item_mapping:\n{mapping}\n"
        f"product_cards:\n{cards}\n"
        "uncertainty:\nUnresolved items remain unsupported."
    )


def _compile_style(
    output: Mapping[str, object] | None,
    policy: DeterministicSemanticPolicy,
) -> str:
    candidates = _sequence(None if output is None else output.get("candidates"))
    cards: list[str] = []
    rationale: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        rendered = _candidate_card(candidate)
        if rendered is None:
            continue
        cards.append(rendered)
        for evidence in _sequence(candidate.get("style_evidence")):
            if not isinstance(evidence, Mapping):
                continue
            reference = _text(evidence.get("evidence_reference"))
            facet = _text(evidence.get("facet"))
            value = _text(evidence.get("value"))
            if None not in (reference, facet, value) and _policy_matches(
                f"{facet} {value}", policy
            ):
                rationale.append(f"{reference} | {facet}: {value}")
    supported = output is not None and output.get("support_status") == "supported"
    if (
        not supported
        or not cards
        or (policy.abstain_when_no_evidence and not rationale)
    ):
        return (
            "answer:\nunable to recommend\n"
            "diversity_rationale:\nnone\n"
            "product_cards:\nnone\n"
            "uncertainty:\nNo supported style evidence was returned by the tool."
        )
    return (
        "answer:\nThe supported style candidates are listed below.\n"
        f"diversity_rationale:\n{'\n'.join(rationale)}\n"
        f"product_cards:\n{'\n'.join(cards)}\n"
        "uncertainty:\nOnly literal returned style facets are used."
    )


def _compile_knowledge(
    output: Mapping[str, object] | None,
    *,
    fallback_marker: str,
    unresolved: str,
    policy: DeterministicSemanticPolicy,
) -> str:
    sources = _sequence(None if output is None else output.get("sources"))
    lines: list[str] = []
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        reference = _text(source.get("evidence_reference"))
        title = _text(source.get("title"))
        body = _text(source.get("text"))
        if None not in (reference, title, body) and _policy_matches(
            f"{title} {body}", policy
        ):
            source_text = f"{title}: {body}"
            segments = tuple(
                segment.strip()
                for segment in _MATERIAL_SEGMENT_RE.split(source_text)
                if _MATERIAL_TEXT_RE.search(segment)
            )
            lines.extend(f"{reference} | {segment}" for segment in segments)
    if not lines:
        return (
            f"answer:\n{fallback_marker}\n"
            f"evidence:\n{fallback_marker}\n"
            f"uncertainty:\n{unresolved}"
        )
    rendered = "\n".join(lines)
    return f"answer:\n{rendered}\nevidence:\n{rendered}\nuncertainty:\n{unresolved}"


def _compile_ocr(
    output: Mapping[str, object] | None,
    policy: DeterministicSemanticPolicy,
) -> str:
    raw_lines = _sequence(None if output is None else output.get("lines"))
    extractions: list[str] = []
    for raw in raw_lines:
        if not isinstance(raw, Mapping):
            continue
        reference = _text(raw.get("line_reference"))
        line_text = _text(raw.get("text"))
        if reference is None or line_text is None:
            continue
        if not _policy_matches(line_text, policy):
            continue
        emitted = False
        for pair in _sequence(raw.get("fields")):
            if (
                isinstance(pair, Sequence)
                and not isinstance(pair, (str, bytes))
                and len(pair) == 2
            ):
                field = _text(pair[0])
                value = _text(pair[1])
                if field is not None and value is not None and value in line_text:
                    extractions.append(f"{field}: {value} {reference}")
                    emitted = True
        if not emitted:
            value = line_text
            if policy.ocr_extraction_plan == "literal-material-spans":
                # Preserve a literal substring while removing terminal sentence
                # punctuation that would separate the line handle from the
                # material statement in the frozen GCS grammar.  Noise-only OCR
                # glyphs are omitted rather than promoted into unsupported facts.
                value = re.sub(r"[。！？.!?]+$", "", value).strip()
                if _MATERIAL_TEXT_RE.search(value) is None:
                    continue
            extractions.append(f"text: {value} {reference}")
    if not extractions:
        return (
            "answer:\nunable to read\n"
            "evidence:\ninsufficient text\n"
            "uncertainty:\nuntrusted document text"
        )
    rendered = "\n".join(extractions[:20])
    return (
        f"answer:\n{rendered}\n"
        f"evidence:\n{rendered}\n"
        "uncertainty:\nuntrusted document text"
    )


def compile_deterministic_response(
    capability_id: str,
    observations: Sequence[DeterministicToolObservation],
    *,
    semantic_policy: DeterministicSemanticPolicy | None = None,
) -> str | None:
    """Compile a response once the capability's deterministic tool path ended."""

    if next_deterministic_tool(capability_id, observations) is not None:
        return None
    policy = semantic_policy or DeterministicSemanticPolicy(  # type: ignore[arg-type]
        capability_id=capability_id
    )
    if policy.capability_id != capability_id:
        raise ValueError("semantic policy capability differs from routed capability")
    _semantic_policy_payload(policy)
    if (
        policy.evidence_terms or policy.ocr_extraction_plan != "all-lines"
    ) and capability_id not in SEMANTIC_POLICY_CAPABILITIES:
        raise ValueError("routed capability does not consume a semantic policy")
    if capability_id == "product.exact_match":
        return _compile_product(_output(observations, "image_product_search"))
    if capability_id == "product.multi_search":
        return _compile_multi(_output(observations, "multi_product_search"))
    if capability_id == "product.style_recommendation":
        return _compile_style(_output(observations, "style_similar_search"), policy)
    if capability_id == "utility.document_reading":
        return _compile_ocr(_output(observations, "document_ocr"), policy)
    if capability_id == "knowledge.visual_encyclopedia":
        return _compile_knowledge(
            _output(observations, "encyclopedia_lookup"),
            fallback_marker="not enough evidence",
            unresolved="The detected identity remains unresolved.",
            policy=policy,
        )
    if capability_id == "utility.recipe_guidance":
        return _compile_knowledge(
            _output(observations, "recipe_lookup"),
            fallback_marker="no supported recipe",
            unresolved="The detected dish identity remains unresolved.",
            policy=policy,
        )
    return None


__all__ = [
    "DETERMINISTIC_ASSISTANT_CONTRACT_SHA256",
    "DETERMINISTIC_ASSISTANT_CONTRACT_VERSION",
    "DETERMINISTIC_SEMANTIC_POLICY_VERSION",
    "SEMANTIC_POLICY_CAPABILITIES",
    "DeterministicSemanticPolicy",
    "DeterministicToolDecision",
    "DeterministicToolObservation",
    "compile_deterministic_response",
    "deterministic_contract_payload",
    "deterministic_tool_names",
    "next_deterministic_tool",
    "parse_deterministic_semantic_policy",
    "render_deterministic_semantic_policy",
]
