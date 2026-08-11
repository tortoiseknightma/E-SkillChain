"""Runner-owned preflight for model-visible Assistant response contracts.

The GCS scorer remains authoritative.  This module is a deliberately smaller
runtime guard which catches response-shape failures before a successful
Assistant result is published.  It uses only model-visible tool output and the
selected runtime capability; no gold answer or scorer sidecar is consulted.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from typing import Literal


ASSISTANT_RESPONSE_PREFLIGHT_POLICY_VERSION = (
    "assistant-response-contract-preflight-v1"
)
ASSISTANT_RESPONSE_REPAIR_POLICY_VERSION = "assistant-response-fixed-repair-v1"

ResponseContractReason = Literal[
    "response_control_token",
    "response_detector_only_final",
    "response_empty",
    "response_fallback_marker_missing",
    "response_repair_new_material_atom",
    "response_section_invalid",
    "response_tool_sequence_invalid",
]

_CONTROL_TOKEN_RE = re.compile(r"<\|[^<>\r\n]{1,128}\|>")
_CONTENT_TOKEN_RE = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:[.,]\d+)*|[\u3400-\u9fff]"
)
_TOP_LEVEL_ASCII_HEADING_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_ -]{0,63}):")
_ENCYCLOPEDIA_CAPABILITY = "knowledge.visual_encyclopedia"
_ENCYCLOPEDIA_ALLOWED_SEQUENCES = frozenset(
    {
        ("encyclopedia_lookup",),
        ("object_detect", "encyclopedia_lookup"),
    }
)
_ENCYCLOPEDIA_FALLBACK_MARKER = "not enough evidence"
_REPAIR_OWNED_HEADINGS = (
    "answer",
    "evidence",
    "uncertainty",
    "product_cards",
    "item_mapping",
    "diversity_rationale",
)
AMBIGUITY_SIGNAL_POLICY = "empty_sources_only_no_reliable_public_ambiguity_field"


@dataclass(frozen=True)
class AssistantResponseToolObservation:
    """One runner-observed tool result, projected exactly as the model saw it."""

    tool_name: str
    status: Literal["success", "error"]
    public_output: Mapping[str, object] | None = None


@dataclass(frozen=True)
class AssistantResponseContractValidation:
    """Deterministic preflight result used before and after one repair."""

    policy_version: str
    valid: bool
    reason_codes: tuple[ResponseContractReason, ...]
    required_sections: tuple[str, ...]
    fallback_marker_required: bool
    ambiguity_signal_policy: Literal[
        "empty_sources_only_no_reliable_public_ambiguity_field"
    ]


def _required_sections(
    contract: Mapping[str, object] | None,
) -> tuple[str, ...]:
    if contract is None:
        return ()
    raw = contract.get("required_sections")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    sections = tuple(item for item in raw if isinstance(item, str) and item)
    return sections if len(sections) == len(raw) else ()


def _sections_are_exact(response_text: str, required: tuple[str, ...]) -> bool:
    """Require exact lowercase ASCII headings, once and in frozen order."""

    if not required or not response_text.startswith(f"{required[0]}:"):
        return False
    lines = response_text.splitlines()
    positions: list[int] = []
    contents: dict[str, list[str]] = {section: [] for section in required}
    current: str | None = None
    for index, raw_line in enumerate(lines):
        matched: str | None = None
        for section in required:
            prefix = f"{section}:"
            if raw_line.startswith(prefix):
                if raw_line[: len(prefix)] != prefix:
                    return False
                matched = section
                positions.append(index)
                current = section
                body = raw_line[len(prefix) :].strip()
                if body:
                    contents[section].append(body)
                break
            # A translated/case-changed required heading must not silently
            # become section content.
            if raw_line.casefold().startswith(prefix.casefold()):
                return False
        if matched is not None:
            continue
        if raw_line.strip():
            if current is None:
                return False
            contents[current].append(raw_line.strip())
    expected_positions = []
    for section in required:
        matches = [
            index
            for index, raw_line in enumerate(lines)
            if raw_line.startswith(f"{section}:")
        ]
        if len(matches) != 1 or not contents[section]:
            return False
        expected_positions.append(matches[0])
    return positions == expected_positions == sorted(expected_positions)


def _has_unexpected_top_level_ascii_heading(
    response_text: str, required: tuple[str, ...]
) -> bool:
    """Detect an extra section marker without rejecting ``field: value`` facts."""

    allowed = set(required)
    for raw_line in response_text.splitlines():
        if raw_line.startswith(("http://", "https://")):
            continue
        match = _TOP_LEVEL_ASCII_HEADING_RE.match(raw_line)
        if (
            match is not None
            and match.group(1) not in allowed
            and not raw_line[match.end() :].strip()
        ):
            return True
    return False


def _encyclopedia_sources_empty(
    observations: Sequence[AssistantResponseToolObservation],
) -> bool:
    lookups = tuple(
        item
        for item in observations
        if item.status == "success" and item.tool_name == "encyclopedia_lookup"
    )
    if not lookups:
        return False
    output = lookups[-1].public_output
    if not isinstance(output, Mapping):
        return True
    sources = output.get("sources")
    return (
        not isinstance(sources, Sequence)
        or isinstance(sources, (str, bytes))
        or not sources
    )


def validate_assistant_response_contract(
    response_text: str,
    *,
    observations: Sequence[AssistantResponseToolObservation] = (),
    selected_capability: str | None = None,
    contract: Mapping[str, object] | None = None,
) -> AssistantResponseContractValidation:
    """Validate the shared response contract without scorer-private inputs."""

    reasons: set[ResponseContractReason] = set()
    if not response_text.strip():
        reasons.add("response_empty")
    if _CONTROL_TOKEN_RE.search(response_text):
        reasons.add("response_control_token")

    names = tuple(item.tool_name for item in observations)
    successful_names = tuple(
        item.tool_name for item in observations if item.status == "success"
    )
    encyclopedia_runtime = (
        selected_capability == _ENCYCLOPEDIA_CAPABILITY
        or "encyclopedia_lookup" in names
    )
    if encyclopedia_runtime and (
        names not in _ENCYCLOPEDIA_ALLOWED_SEQUENCES
        or any(item.status != "success" for item in observations)
    ):
        reasons.add("response_tool_sequence_invalid")
    if successful_names and successful_names[-1] == "object_detect":
        reasons.add("response_detector_only_final")

    required = _required_sections(contract)
    if encyclopedia_runtime and not required:
        required = ("answer", "evidence", "uncertainty")
    if required and (
        not _sections_are_exact(response_text, required)
        or _has_unexpected_top_level_ascii_heading(response_text, required)
    ):
        reasons.add("response_section_invalid")

    successful_lookup_present = any(
        item.status == "success" and item.tool_name == "encyclopedia_lookup"
        for item in observations
    )
    fallback_required = encyclopedia_runtime and (
        not successful_lookup_present
        or _encyclopedia_sources_empty(observations)
    )
    if (
        fallback_required
        and _ENCYCLOPEDIA_FALLBACK_MARKER not in response_text
    ):
        reasons.add("response_fallback_marker_missing")

    ordered = tuple(sorted(reasons))
    return AssistantResponseContractValidation(
        policy_version=ASSISTANT_RESPONSE_PREFLIGHT_POLICY_VERSION,
        valid=not ordered,
        reason_codes=ordered,
        required_sections=required,
        fallback_marker_required=fallback_required,
        ambiguity_signal_policy=AMBIGUITY_SIGNAL_POLICY,
    )


def _repair_content_tokens(text: str) -> Counter[str]:
    """Remove runner-owned syntax, then retain a word/CJK token multiset."""

    cleaned = _CONTROL_TOKEN_RE.sub(" ", text)
    cleaned = re.sub(
        re.escape(_ENCYCLOPEDIA_FALLBACK_MARKER),
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    normalized_lines: list[str] = []
    for raw_line in cleaned.splitlines():
        line = raw_line
        for heading in _REPAIR_OWNED_HEADINGS:
            prefix = f"{heading}:"
            if line.startswith(prefix):
                line = line[len(prefix) :]
                break
        normalized_lines.append(line)
    return Counter(
        match.group(0).casefold()
        for match in _CONTENT_TOKEN_RE.finditer("\n".join(normalized_lines))
    )


def repair_preserves_material_atoms(original: str, repaired: str) -> bool:
    """Prove every repaired content token came from the original draft.

    Only runner-owned headings, control-token deletion, and the exact frozen
    Encyclopedia fallback marker are exempt.  A normal entity or factual word
    therefore cannot be introduced by the format-only model call.
    """

    return _repair_content_tokens(repaired) <= _repair_content_tokens(original)


def fixed_response_repair_prompt(
    *,
    validation: AssistantResponseContractValidation,
) -> str:
    """Return the one fixed, no-tool, no-new-facts repair instruction."""

    sections = ", ".join(f"`{item}:`" for item in validation.required_sections)
    structure = (
        f"Use these exact lowercase ASCII headings once and in this order: {sections}. "
        "Move existing text beneath them; every section must be non-empty."
        if sections
        else "Preserve the existing plain-text structure after removing control tokens."
    )
    marker = (
        " Add the exact standalone phrase `not enough evidence` where required."
        if validation.fallback_marker_required
        else ""
    )
    return (
        "One-time fixed response-format repair. Reformat only the immediately "
        "preceding draft using the already-visible tool results. Do not call a "
        "tool, inspect the image again, add a fact, number, URL, product handle, "
        "or evidence handle, and do not paraphrase factual text. Remove every "
        "model control token. Return only the repaired final answer. "
        f"{structure}"
        f"{marker}"
    )


__all__ = [
    "ASSISTANT_RESPONSE_PREFLIGHT_POLICY_VERSION",
    "ASSISTANT_RESPONSE_REPAIR_POLICY_VERSION",
    "AMBIGUITY_SIGNAL_POLICY",
    "AssistantResponseContractValidation",
    "AssistantResponseToolObservation",
    "fixed_response_repair_prompt",
    "repair_preserves_material_atoms",
    "validate_assistant_response_contract",
]
