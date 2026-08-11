"""Deterministic, text-only Hybrid Router training and offline evaluation.

This module deliberately stops at an offline routing candidate.  It cannot
call a model, inspect an image, or modify the assistant runtime.  The only
learned feature is the ordered projection of user-turn text.
"""

from __future__ import annotations

import io
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import shutil
import stat
from typing import Literal, Self

import joblib
import sklearn
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from skillchain.schemas import ConversationTurn, Query
from skillchain.synthesis.store import (
    atomic_publish_new_directory,
    new_staging_directory,
)
from skillchain.tools.serialization import (
    canonical_json_bytes,
    read_stable_regular_file,
    sha256_bytes,
)


CAPABILITY_ORDER: tuple[str, ...] = (
    "product.exact_match",
    "product.multi_search",
    "product.style_recommendation",
    "knowledge.visual_encyclopedia",
    "utility.document_reading",
    "utility.recipe_guidance",
)
FOLD_COUNT = 4
FOLD_IDS: tuple[str, ...] = tuple(
    f"fold-{index:02d}" for index in range(FOLD_COUNT)
)
DEFAULT_RANDOM_SEED = 20260808
AUTHORIZATION_STATUS = "routing_only_not_runtime_authorized"

FEATURE_CONTRACT: dict[str, object] = {
    "projection": "ordered_user_turn_content_joined_by_newline",
    "excluded": [
        "assistant_turns",
        "assets",
        "images",
        "labels",
        "grouping_fields",
        "split",
    ],
    "vectorizer": {
        "analyzer": "char",
        "ngram_range": [2, 5],
        "lowercase": True,
        "norm": "l2",
        "min_df": 1,
    },
}
TRAINING_CONTRACT: dict[str, object] = {
    "estimator": "sklearn.linear_model.LogisticRegression",
    "class_weight": "balanced",
    "solver": "lbfgs",
    "C": 1.0,
    "max_iter": 2000,
    "tol": 0.0001,
    "fold_count": FOLD_COUNT,
    "capability_order": list(CAPABILITY_ORDER),
    "probability_tie_break": "first_in_capability_order",
}


def _hash_payload(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


FEATURE_CONTRACT_SHA256 = _hash_payload(FEATURE_CONTRACT)
TRAINING_CONTRACT_SHA256 = _hash_payload(TRAINING_CONTRACT)


class HybridRouterError(ValueError):
    """Raised when router inputs, evidence, or artifacts fail closed."""


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _trimmed(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise HybridRouterError(f"{label} must be non-blank and trimmed")
    return value


def _sha256(value: str, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise HybridRouterError(f"{label} must be a lowercase SHA-256")
    return value


def _turn_parts(turn: ConversationTurn | Mapping[str, object]) -> tuple[str, str]:
    if isinstance(turn, ConversationTurn):
        return turn.role, turn.content
    if not isinstance(turn, Mapping) or set(turn) != {"role", "content"}:
        raise HybridRouterError("each turn must contain exactly role and content")
    role = turn["role"]
    content = turn["content"]
    if role not in {"user", "assistant"} or not isinstance(content, str):
        raise HybridRouterError("turn role/content is invalid")
    content = content.strip()
    if not content:
        raise HybridRouterError("turn content must be non-blank")
    return role, content


def project_user_turns(
    turns: Iterable[ConversationTurn | Mapping[str, object]],
) -> str:
    """Return the entire and only model feature: ordered user-turn content."""

    user_content: list[str] = []
    saw_turn = False
    for turn in turns:
        saw_turn = True
        role, content = _turn_parts(turn)
        if role == "user":
            user_content.append(content)
    if not saw_turn or not user_content:
        raise HybridRouterError("router input must contain at least one user turn")
    return "\n".join(user_content)


class HybridRouteExample(_FrozenModel):
    """Minimal training row after the public-input projection boundary."""

    query_id: str
    user_text: str
    canonical_capability: str
    fold_id: str
    baseline_capability: str | None = None
    baseline_observed: bool

    @field_validator("query_id", "user_text", "fold_id")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _trimmed(value, info.field_name)

    @field_validator("canonical_capability")
    @classmethod
    def validate_label(cls, value: str) -> str:
        if value not in CAPABILITY_ORDER:
            raise ValueError("canonical_capability is outside the frozen enum")
        return value

    @field_validator("baseline_capability")
    @classmethod
    def validate_baseline(cls, value: str | None) -> str | None:
        if value is not None and value not in CAPABILITY_ORDER:
            raise ValueError("baseline_capability is outside the frozen enum")
        return value


def make_hybrid_route_example(
    query: Query | Mapping[str, object],
    *,
    fold_id: str,
    baseline_capability: str | None = None,
    baseline_observed: bool = False,
) -> HybridRouteExample:
    """Project a Query without exposing non-user fields to the vectorizer."""

    if isinstance(query, Query):
        query_id = query.query_id
        turns: Iterable[ConversationTurn | Mapping[str, object]] = query.turns
        canonical_capability = query.canonical_capability
    elif isinstance(query, Mapping):
        query_id = query.get("query_id")
        turns_value = query.get("turns")
        canonical_capability = query.get("canonical_capability")
        if not isinstance(turns_value, (list, tuple)):
            raise HybridRouterError("query turns must be a list or tuple")
        turns = turns_value
    else:
        raise TypeError("query must be a Query or mapping")
    if not isinstance(query_id, str):
        raise HybridRouterError("query_id must be text")
    if not isinstance(canonical_capability, str):
        raise HybridRouterError("canonical_capability must be resolved")
    return HybridRouteExample(
        query_id=query_id,
        user_text=project_user_turns(turns),
        canonical_capability=canonical_capability,
        fold_id=fold_id,
        baseline_capability=baseline_capability,
        baseline_observed=baseline_observed,
    )


class HybridRoutePrediction(_FrozenModel):
    query_id: str
    fold_id: str
    expected_capability: str
    baseline_capability: str | None
    baseline_observed: bool
    local_capability: str
    probabilities: tuple[float, ...]
    top1_probability: float
    top2_margin: float

    @field_validator("probabilities", mode="before")
    @classmethod
    def coerce_probabilities(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_prediction(self) -> Self:
        _trimmed(self.query_id, "query_id")
        _trimmed(self.fold_id, "fold_id")
        if self.expected_capability not in CAPABILITY_ORDER:
            raise ValueError("expected_capability is outside the frozen enum")
        if self.baseline_capability is not None and self.baseline_capability not in CAPABILITY_ORDER:
            raise ValueError("baseline_capability is outside the frozen enum")
        if self.local_capability not in CAPABILITY_ORDER:
            raise ValueError("local_capability is outside the frozen enum")
        if len(self.probabilities) != len(CAPABILITY_ORDER):
            raise ValueError("probabilities do not match the frozen class order")
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in self.probabilities):
            raise ValueError("probabilities must be finite values in [0, 1]")
        if not math.isclose(sum(self.probabilities), 1.0, abs_tol=1e-7):
            raise ValueError("probabilities must sum to one")
        ranked = sorted(
            range(len(CAPABILITY_ORDER)),
            key=lambda index: (-self.probabilities[index], index),
        )
        expected_local = CAPABILITY_ORDER[ranked[0]]
        expected_top1 = self.probabilities[ranked[0]]
        expected_margin = expected_top1 - self.probabilities[ranked[1]]
        if self.local_capability != expected_local:
            raise ValueError("local_capability violates frozen probability tie-break")
        if not math.isclose(self.top1_probability, expected_top1, abs_tol=1e-12):
            raise ValueError("top1_probability does not match probabilities")
        if not math.isclose(self.top2_margin, expected_margin, abs_tol=1e-12):
            raise ValueError("top2_margin does not match probabilities")
        return self


class HybridRouteMetrics(_FrozenModel):
    query_count: int
    local_accepted_count: int
    fallback_count: int
    coverage: float
    query_accuracy: float
    micro_f1: float
    macro_f1: float
    recall_by_capability: dict[str, float]
    f1_by_capability: dict[str, float]
    paired_comparison_available: bool
    corrected: int | None
    broken: int | None
    utility: float | None

    @model_validator(mode="after")
    def validate_paired_metrics(self) -> Self:
        paired = (self.corrected, self.broken, self.utility)
        if self.paired_comparison_available:
            if any(value is None for value in paired):
                raise ValueError("available paired metrics may not be null")
        elif any(value is not None for value in paired):
            raise ValueError("unavailable paired metrics must be null")
        return self


class HybridOOFResult(_FrozenModel):
    schema_version: Literal[1] = 1
    authorization_status: Literal["routing_only_not_runtime_authorized"] = (
        AUTHORIZATION_STATUS
    )
    random_seed: int
    fold_ids: tuple[str, ...]
    examples_sha256: str
    fold_mapping_sha256: str
    predictions: tuple[HybridRoutePrediction, ...]
    metrics: HybridRouteMetrics

    @field_validator("fold_ids", mode="before")
    @classmethod
    def coerce_fold_ids(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("predictions", mode="before")
    @classmethod
    def coerce_predictions(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value


class HybridThresholdPolicy(_FrozenModel):
    probability_threshold: float
    margin_threshold: float
    authorization_status: Literal["routing_only_not_runtime_authorized"] = (
        AUTHORIZATION_STATUS
    )
    metrics: HybridRouteMetrics
    macro_f1_gain: float
    minimum_capability_recall_delta: float


class HybridThresholdSelection(_FrozenModel):
    outcome: Literal["candidate", "no_candidate"]
    baseline_metrics: HybridRouteMetrics
    evaluated_candidate_count: int
    policy: HybridThresholdPolicy | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> Self:
        if (self.outcome == "candidate") != (self.policy is not None):
            raise ValueError("threshold outcome and policy disagree")
        return self


class HybridRouteDecision(_FrozenModel):
    outcome: Literal["local_route", "fallback_required"]
    selected_capability: str | None = None

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        if self.outcome == "local_route":
            if self.selected_capability not in CAPABILITY_ORDER:
                raise ValueError("local route must select a frozen capability")
        elif self.selected_capability is not None:
            raise ValueError("fallback_required may not select a capability")
        return self


def build_hybrid_router(*, random_seed: int = DEFAULT_RANDOM_SEED) -> Pipeline:
    """Construct the frozen sklearn Pipeline without fitting it."""

    return Pipeline(
        steps=(
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 5),
                    lowercase=True,
                    norm="l2",
                    min_df=1,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    solver="lbfgs",
                    C=1.0,
                    max_iter=2000,
                    tol=1e-4,
                    random_state=random_seed,
                ),
            ),
        )
    )


def _ordered_examples(
    examples: Sequence[HybridRouteExample], *, require_four_folds: bool
) -> tuple[HybridRouteExample, ...]:
    if not examples:
        raise HybridRouterError("Hybrid Router requires at least one example")
    verified: list[HybridRouteExample] = []
    for example in examples:
        if type(example) is not HybridRouteExample:
            raise TypeError("every example must be a HybridRouteExample")
        verified.append(
            HybridRouteExample.model_validate(example.model_dump(mode="python"))
        )
    ordered = tuple(sorted(verified, key=lambda item: item.query_id))
    if len({item.query_id for item in ordered}) != len(ordered):
        raise HybridRouterError("Hybrid Router query IDs must be unique")
    if {item.canonical_capability for item in ordered} != set(CAPABILITY_ORDER):
        raise HybridRouterError("examples must cover the frozen six capabilities")
    fold_ids = tuple(sorted({item.fold_id for item in ordered}))
    if require_four_folds and fold_ids != FOLD_IDS:
        raise HybridRouterError(
            f"OOF requires exactly the frozen folds {FOLD_IDS}"
        )
    return ordered


def _examples_sha256(examples: Sequence[HybridRouteExample]) -> str:
    return _hash_payload([item.model_dump(mode="json") for item in examples])


def _fold_mapping_sha256(examples: Sequence[HybridRouteExample]) -> str:
    return _hash_payload(
        [{"query_id": item.query_id, "fold_id": item.fold_id} for item in examples]
    )


def _validate_fitted_pipeline(
    pipeline: Pipeline, *, expected_random_seed: int | None = None
) -> None:
    if type(pipeline) is not Pipeline:
        raise HybridRouterError("router artifact must contain an sklearn Pipeline")
    if tuple(pipeline.named_steps) != ("tfidf", "classifier"):
        raise HybridRouterError("router Pipeline step contract drifted")
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    if type(vectorizer) is not TfidfVectorizer or type(classifier) is not LogisticRegression:
        raise HybridRouterError("router Pipeline component type drifted")
    pipeline_params = pipeline.get_params(deep=False)
    for key, expected in {
        "memory": None,
        "transform_input": None,
        "verbose": False,
    }.items():
        if pipeline_params.get(key) != expected:
            raise HybridRouterError(f"router Pipeline parameter drifted: {key}")
    vectorizer_params = vectorizer.get_params(deep=False)
    classifier_params = classifier.get_params(deep=False)
    observed_random_seed = classifier_params.get("random_state")
    if (
        not isinstance(observed_random_seed, int)
        or isinstance(observed_random_seed, bool)
        or observed_random_seed < 0
    ):
        raise HybridRouterError("router classifier random seed is invalid")
    reference_seed = (
        observed_random_seed
        if expected_random_seed is None
        else expected_random_seed
    )
    reference = build_hybrid_router(random_seed=reference_seed)
    expected_vectorizer = reference.named_steps["tfidf"].get_params(deep=False)
    for key, expected in expected_vectorizer.items():
        if vectorizer_params.get(key) != expected:
            raise HybridRouterError(f"router vectorizer parameter drifted: {key}")
    expected_classifier = reference.named_steps["classifier"].get_params(deep=False)
    for key, expected in expected_classifier.items():
        if classifier_params.get(key) != expected:
            raise HybridRouterError(f"router classifier parameter drifted: {key}")
    if not hasattr(vectorizer, "vocabulary_") or not hasattr(classifier, "classes_"):
        raise HybridRouterError("router Pipeline is not fitted")
    if set(classifier.classes_.tolist()) != set(CAPABILITY_ORDER):
        raise HybridRouterError("router fitted classes differ from frozen capability enum")


def fit_hybrid_router(
    examples: Sequence[HybridRouteExample], *, random_seed: int = DEFAULT_RANDOM_SEED
) -> Pipeline:
    """Fit one candidate on all supplied opt examples."""

    ordered = _ordered_examples(examples, require_four_folds=False)
    pipeline = build_hybrid_router(random_seed=random_seed)
    pipeline.fit(
        [item.user_text for item in ordered],
        [item.canonical_capability for item in ordered],
    )
    _validate_fitted_pipeline(pipeline)
    return pipeline


def _predict_rows(
    pipeline: Pipeline, examples: Sequence[HybridRouteExample]
) -> tuple[HybridRoutePrediction, ...]:
    _validate_fitted_pipeline(pipeline)
    classifier = pipeline.named_steps["classifier"]
    class_indexes = {
        capability: index for index, capability in enumerate(classifier.classes_.tolist())
    }
    raw_probabilities = pipeline.predict_proba([item.user_text for item in examples])
    predictions: list[HybridRoutePrediction] = []
    for example, raw in zip(examples, raw_probabilities, strict=True):
        probabilities = tuple(
            float(raw[class_indexes[capability]]) for capability in CAPABILITY_ORDER
        )
        ranked = sorted(
            range(len(CAPABILITY_ORDER)),
            key=lambda index: (-probabilities[index], index),
        )
        top1 = probabilities[ranked[0]]
        predictions.append(
            HybridRoutePrediction(
                query_id=example.query_id,
                fold_id=example.fold_id,
                expected_capability=example.canonical_capability,
                baseline_capability=example.baseline_capability,
                baseline_observed=example.baseline_observed,
                local_capability=CAPABILITY_ORDER[ranked[0]],
                probabilities=probabilities,
                top1_probability=top1,
                top2_margin=top1 - probabilities[ranked[1]],
            )
        )
    return tuple(predictions)


def local_takeover(
    prediction: HybridRoutePrediction,
    *,
    probability_threshold: float,
    margin_threshold: float,
) -> bool:
    """Return the frozen AND decision for local routing versus fallback."""

    for value, label in (
        (probability_threshold, "probability_threshold"),
        (margin_threshold, "margin_threshold"),
    ):
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise HybridRouterError(f"{label} must be a finite value in [0, 1]")
    return (
        prediction.top1_probability >= probability_threshold
        and prediction.top2_margin >= margin_threshold
    )


def decide_hybrid_route(
    prediction: HybridRoutePrediction,
    *,
    probability_threshold: float,
    margin_threshold: float,
) -> HybridRouteDecision:
    """Return a local capability or an explicit request for the paired fallback."""

    if local_takeover(
        prediction,
        probability_threshold=probability_threshold,
        margin_threshold=margin_threshold,
    ):
        return HybridRouteDecision(
            outcome="local_route", selected_capability=prediction.local_capability
        )
    return HybridRouteDecision(outcome="fallback_required")


def _classification(
    expected: Sequence[str], selected: Sequence[str | None]
) -> tuple[float, float, dict[str, float], dict[str, float]]:
    correct = sum(left == right for left, right in zip(expected, selected, strict=True))
    accuracy = correct / len(expected)
    recalls: dict[str, float] = {}
    f1s: dict[str, float] = {}
    for capability in CAPABILITY_ORDER:
        tp = sum(left == capability and right == capability for left, right in zip(expected, selected, strict=True))
        fp = sum(left != capability and right == capability for left, right in zip(expected, selected, strict=True))
        fn = sum(left == capability and right != capability for left, right in zip(expected, selected, strict=True))
        recalls[capability] = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = recalls[capability]
        f1s[capability] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # For this closed-world, single-label task, micro-F1 is query accuracy.
    # ``None`` is an unresolved route and is therefore an ordinary wrong row.
    return accuracy, accuracy, recalls, f1s


def _evaluate_with_acceptance(
    predictions: Sequence[HybridRoutePrediction], accepted: Sequence[bool]
) -> HybridRouteMetrics:
    if not predictions:
        raise HybridRouterError("metrics require at least one prediction")
    if len(predictions) != len(accepted):
        raise HybridRouterError("acceptance mask length differs from predictions")
    expected = [item.expected_capability for item in predictions]
    selected = [
        item.local_capability if use_local else item.baseline_capability
        for item, use_local in zip(predictions, accepted, strict=True)
    ]
    query_accuracy, micro_f1, recalls, f1s = _classification(expected, selected)
    paired_comparison_available = all(
        item.baseline_observed for item in predictions
    )
    corrected: int | None = None
    broken: int | None = None
    utility: float | None = None
    if paired_comparison_available:
        corrected = 0
        broken = 0
        for item, after in zip(predictions, selected, strict=True):
            before_ok = item.baseline_capability == item.expected_capability
            after_ok = after == item.expected_capability
            corrected += int(not before_ok and after_ok)
            broken += int(before_ok and not after_ok)
        utility = corrected - 2.5 * broken
    accepted_count = sum(accepted)
    return HybridRouteMetrics(
        query_count=len(predictions),
        local_accepted_count=accepted_count,
        fallback_count=len(predictions) - accepted_count,
        coverage=accepted_count / len(predictions),
        query_accuracy=query_accuracy,
        micro_f1=micro_f1,
        macro_f1=sum(f1s.values()) / len(CAPABILITY_ORDER),
        recall_by_capability=recalls,
        f1_by_capability=f1s,
        paired_comparison_available=paired_comparison_available,
        corrected=corrected,
        broken=broken,
        utility=utility,
    )


def evaluate_hybrid_predictions(
    predictions: Sequence[HybridRoutePrediction],
    *,
    probability_threshold: float = 0.0,
    margin_threshold: float = 0.0,
) -> HybridRouteMetrics:
    """Evaluate the hybrid route; low-confidence rows use the paired fallback."""

    ordered = tuple(sorted(predictions, key=lambda item: item.query_id))
    if len({item.query_id for item in ordered}) != len(ordered):
        raise HybridRouterError("prediction query IDs must be unique")
    accepted = tuple(
        local_takeover(
            item,
            probability_threshold=probability_threshold,
            margin_threshold=margin_threshold,
        )
        for item in ordered
    )
    return _evaluate_with_acceptance(ordered, accepted)


def run_grouped_oof(
    examples: Sequence[HybridRouteExample], *, random_seed: int = DEFAULT_RANDOM_SEED
) -> HybridOOFResult:
    """Run deterministic four-fold OOF training using preassigned group-safe folds."""

    ordered = _ordered_examples(examples, require_four_folds=True)
    fold_ids = tuple(sorted({item.fold_id for item in ordered}))
    predictions: list[HybridRoutePrediction] = []
    for fold_id in fold_ids:
        training = tuple(item for item in ordered if item.fold_id != fold_id)
        held_out = tuple(item for item in ordered if item.fold_id == fold_id)
        if not held_out:
            raise HybridRouterError(f"OOF fold is empty: {fold_id}")
        if {item.canonical_capability for item in training} != set(CAPABILITY_ORDER):
            raise HybridRouterError(f"OOF training partition lacks a capability: {fold_id}")
        pipeline = fit_hybrid_router(training, random_seed=random_seed)
        predictions.extend(_predict_rows(pipeline, held_out))
    ordered_predictions = tuple(sorted(predictions, key=lambda item: item.query_id))
    if tuple(item.query_id for item in ordered_predictions) != tuple(
        item.query_id for item in ordered
    ):
        raise HybridRouterError("OOF did not produce exactly one row per input query")
    metrics = evaluate_hybrid_predictions(ordered_predictions)
    return HybridOOFResult(
        random_seed=random_seed,
        fold_ids=fold_ids,
        examples_sha256=_examples_sha256(ordered),
        fold_mapping_sha256=_fold_mapping_sha256(ordered),
        predictions=ordered_predictions,
        metrics=metrics,
    )


def select_threshold_policy(
    predictions: Sequence[HybridRoutePrediction],
    *,
    probability_thresholds: Sequence[float],
    margin_thresholds: Sequence[float],
    min_coverage: float = 0.70,
    min_macro_f1_gain: float = 0.03,
    max_capability_recall_drop: float = 0.03,
) -> HybridThresholdSelection:
    """Select a route-gate candidate without authorizing runtime integration."""

    ordered = tuple(sorted(predictions, key=lambda item: item.query_id))
    if not ordered:
        raise HybridRouterError("threshold selection requires predictions")
    if not all(item.baseline_observed for item in ordered):
        raise HybridRouterError(
            "threshold selection requires an observed paired fallback for every row"
        )
    if {item.expected_capability for item in ordered} != set(CAPABILITY_ORDER):
        raise HybridRouterError(
            "threshold selection requires all six frozen capabilities"
        )
    if not 0.0 <= min_coverage <= 1.0:
        raise HybridRouterError("min_coverage must be in [0, 1]")
    if min_macro_f1_gain < 0.0 or max_capability_recall_drop < 0.0:
        raise HybridRouterError("threshold constraints must be non-negative")
    probability_grid = tuple(sorted(set(probability_thresholds)))
    margin_grid = tuple(sorted(set(margin_thresholds)))
    if not probability_grid or not margin_grid:
        raise HybridRouterError("threshold grids must be non-empty")
    # Validate every grid value, including values that would not be selected.
    for probability in probability_grid:
        for margin in margin_grid:
            local_takeover(
                ordered[0],
                probability_threshold=probability,
                margin_threshold=margin,
            )

    baseline = _evaluate_with_acceptance(ordered, (False,) * len(ordered))
    eligible: list[HybridThresholdPolicy] = []
    for probability in probability_grid:
        for margin in margin_grid:
            metrics = evaluate_hybrid_predictions(
                ordered,
                probability_threshold=probability,
                margin_threshold=margin,
            )
            assert metrics.utility is not None
            macro_gain = metrics.macro_f1 - baseline.macro_f1
            minimum_recall_delta = min(
                metrics.recall_by_capability[capability]
                - baseline.recall_by_capability[capability]
                for capability in CAPABILITY_ORDER
            )
            if (
                metrics.coverage >= min_coverage
                and macro_gain >= min_macro_f1_gain
                and minimum_recall_delta >= -max_capability_recall_drop
                and metrics.utility > 0.0
            ):
                eligible.append(
                    HybridThresholdPolicy(
                        probability_threshold=probability,
                        margin_threshold=margin,
                        metrics=metrics,
                        macro_f1_gain=macro_gain,
                        minimum_capability_recall_delta=minimum_recall_delta,
                    )
                )
    if not eligible:
        return HybridThresholdSelection(
            outcome="no_candidate",
            baseline_metrics=baseline,
            evaluated_candidate_count=len(probability_grid) * len(margin_grid),
        )
    # Utility is the primary objective. Remaining keys make exact ties stable and
    # prefer broader coverage, followed by the least restrictive thresholds.
    chosen = max(
        eligible,
        key=lambda item: (
            item.metrics.utility,
            item.macro_f1_gain,
            item.metrics.coverage,
            -item.probability_threshold,
            -item.margin_threshold,
        ),
    )
    return HybridThresholdSelection(
        outcome="candidate",
        baseline_metrics=baseline,
        evaluated_candidate_count=len(probability_grid) * len(margin_grid),
        policy=chosen,
    )


class HybridRouterManifest(_FrozenModel):
    schema_version: Literal[1] = 1
    authorization_status: Literal["routing_only_not_runtime_authorized"] = (
        AUTHORIZATION_STATUS
    )
    capability_order: tuple[str, ...]
    source_queries_sha256: str
    fold_plan_sha256: str
    oof_examples_sha256: str
    oof_fold_mapping_sha256: str
    feature_contract_sha256: str
    training_contract_sha256: str
    random_seed: int
    model_file: Literal["router.joblib"] = "router.joblib"
    model_sha256: str
    oof_rows_file: Literal["oof-rows.jsonl"] = "oof-rows.jsonl"
    oof_rows_sha256: str
    cv_report_file: Literal["cv-report.json"] = "cv-report.json"
    cv_report_sha256: str
    sklearn_version: str
    joblib_version: str
    manifest_sha256: str

    @field_validator("capability_order", mode="before")
    @classmethod
    def coerce_capabilities(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if self.capability_order != CAPABILITY_ORDER:
            raise ValueError("manifest capability order drifted")
        for field_name in (
            "source_queries_sha256",
            "fold_plan_sha256",
            "oof_examples_sha256",
            "oof_fold_mapping_sha256",
            "feature_contract_sha256",
            "training_contract_sha256",
            "model_sha256",
            "oof_rows_sha256",
            "cv_report_sha256",
            "manifest_sha256",
        ):
            _sha256(getattr(self, field_name), field_name)
        if self.feature_contract_sha256 != FEATURE_CONTRACT_SHA256:
            raise ValueError("manifest feature contract drifted")
        if self.training_contract_sha256 != TRAINING_CONTRACT_SHA256:
            raise ValueError("manifest training contract drifted")
        expected = _hash_payload(
            self.model_dump(mode="json", exclude={"manifest_sha256"})
        )
        if self.manifest_sha256 != expected:
            raise ValueError("router manifest self hash mismatch")
        return self


class LoadedHybridRouterBundle:
    """Verified in-memory contents of a published offline bundle."""

    def __init__(
        self,
        *,
        pipeline: Pipeline,
        manifest: HybridRouterManifest,
        predictions: tuple[HybridRoutePrediction, ...],
        metrics: HybridRouteMetrics,
    ) -> None:
        self.pipeline = pipeline
        self.manifest = manifest
        self.predictions = predictions
        self.metrics = metrics


def _verify_oof_result(result: HybridOOFResult) -> HybridOOFResult:
    try:
        verified = HybridOOFResult.model_validate(result.model_dump(mode="python"))
    except Exception as error:
        raise HybridRouterError("OOF result is invalid") from error
    predictions = tuple(sorted(verified.predictions, key=lambda item: item.query_id))
    if predictions != verified.predictions or len({item.query_id for item in predictions}) != len(predictions):
        raise HybridRouterError("OOF predictions must be unique and query-sorted")
    if tuple(sorted({item.fold_id for item in predictions})) != verified.fold_ids:
        raise HybridRouterError("OOF fold IDs disagree with predictions")
    if len(verified.fold_ids) != FOLD_COUNT:
        raise HybridRouterError("OOF result must contain four folds")
    if evaluate_hybrid_predictions(predictions) != verified.metrics:
        raise HybridRouterError("OOF metrics disagree with prediction rows")
    return verified


def _jsonl_bytes(predictions: Sequence[HybridRoutePrediction]) -> bytes:
    return b"".join(
        canonical_json_bytes(item.model_dump(mode="json")) for item in predictions
    )


def publish_hybrid_router_bundle(
    output_directory: str | Path,
    *,
    pipeline: Pipeline,
    oof_result: HybridOOFResult,
    source_queries_sha256: str,
    fold_plan_sha256: str,
) -> HybridRouterManifest:
    """Create one immutable offline bundle; an existing destination is refused."""

    _sha256(source_queries_sha256, "source_queries_sha256")
    _sha256(fold_plan_sha256, "fold_plan_sha256")
    oof = _verify_oof_result(oof_result)
    _validate_fitted_pipeline(pipeline, expected_random_seed=oof.random_seed)
    destination = Path(output_directory)
    staging = new_staging_directory(destination)
    try:
        model_path = staging / "router.joblib"
        joblib.dump(pipeline, model_path, compress=0)
        model_bytes = model_path.read_bytes()
        oof_bytes = _jsonl_bytes(oof.predictions)
        report_payload = {
            "schema_version": 1,
            "authorization_status": AUTHORIZATION_STATUS,
            "random_seed": oof.random_seed,
            "fold_ids": list(oof.fold_ids),
            "examples_sha256": oof.examples_sha256,
            "fold_mapping_sha256": oof.fold_mapping_sha256,
            "feature_contract": FEATURE_CONTRACT,
            "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
            "training_contract": TRAINING_CONTRACT,
            "training_contract_sha256": TRAINING_CONTRACT_SHA256,
            "metrics": oof.metrics.model_dump(mode="json"),
        }
        report_bytes = canonical_json_bytes(report_payload)
        (staging / "oof-rows.jsonl").write_bytes(oof_bytes)
        (staging / "cv-report.json").write_bytes(report_bytes)
        unsigned_manifest = {
            "schema_version": 1,
            "authorization_status": AUTHORIZATION_STATUS,
            "capability_order": list(CAPABILITY_ORDER),
            "source_queries_sha256": source_queries_sha256,
            "fold_plan_sha256": fold_plan_sha256,
            "oof_examples_sha256": oof.examples_sha256,
            "oof_fold_mapping_sha256": oof.fold_mapping_sha256,
            "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
            "training_contract_sha256": TRAINING_CONTRACT_SHA256,
            "random_seed": oof.random_seed,
            "model_file": "router.joblib",
            "model_sha256": sha256_bytes(model_bytes),
            "oof_rows_file": "oof-rows.jsonl",
            "oof_rows_sha256": sha256_bytes(oof_bytes),
            "cv_report_file": "cv-report.json",
            "cv_report_sha256": sha256_bytes(report_bytes),
            "sklearn_version": sklearn.__version__,
            "joblib_version": joblib.__version__,
        }
        manifest_payload = {
            **unsigned_manifest,
            "manifest_sha256": _hash_payload(unsigned_manifest),
        }
        manifest = HybridRouterManifest.model_validate(manifest_payload)
        (staging / "model-manifest.json").write_bytes(
            canonical_json_bytes(manifest.model_dump(mode="json"))
        )
        atomic_publish_new_directory(staging, destination)
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _read_bundle_file(root: Path, name: str, *, max_bytes: int) -> bytes:
    return read_stable_regular_file(root / name, label=name, max_bytes=max_bytes)


def load_hybrid_router_bundle(
    output_directory: str | Path,
    *,
    expected_manifest_file_sha256: str,
    expected_source_queries_sha256: str | None = None,
    expected_fold_plan_sha256: str | None = None,
) -> LoadedHybridRouterBundle:
    """Load only after validating an external manifest trust anchor.

    ``router.joblib`` is a pickle-backed artifact.  The caller-supplied
    manifest file digest is therefore mandatory and is checked before the
    manifest is parsed or any joblib bytes are deserialized.
    """

    root = Path(output_directory)
    _sha256(expected_manifest_file_sha256, "expected_manifest_file_sha256")
    try:
        metadata = root.lstat()
    except OSError as error:
        raise HybridRouterError("router bundle directory cannot be inspected") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise HybridRouterError("router bundle must be a real directory")
    manifest_bytes = _read_bundle_file(root, "model-manifest.json", max_bytes=128_000)
    if sha256_bytes(manifest_bytes) != expected_manifest_file_sha256:
        raise HybridRouterError("router manifest file hash differs from expected")
    try:
        manifest = HybridRouterManifest.model_validate_json(manifest_bytes)
    except Exception as error:
        raise HybridRouterError("router manifest is invalid") from error
    if expected_source_queries_sha256 is not None and manifest.source_queries_sha256 != expected_source_queries_sha256:
        raise HybridRouterError("router source query hash differs from expected")
    if expected_fold_plan_sha256 is not None and manifest.fold_plan_sha256 != expected_fold_plan_sha256:
        raise HybridRouterError("router fold plan hash differs from expected")
    if manifest.sklearn_version != sklearn.__version__ or manifest.joblib_version != joblib.__version__:
        raise HybridRouterError("router dependency version drifted")

    model_bytes = _read_bundle_file(root, manifest.model_file, max_bytes=512_000_000)
    oof_bytes = _read_bundle_file(root, manifest.oof_rows_file, max_bytes=64_000_000)
    report_bytes = _read_bundle_file(root, manifest.cv_report_file, max_bytes=2_000_000)
    observed_hashes = {
        "model": sha256_bytes(model_bytes),
        "oof": sha256_bytes(oof_bytes),
        "report": sha256_bytes(report_bytes),
    }
    if observed_hashes != {
        "model": manifest.model_sha256,
        "oof": manifest.oof_rows_sha256,
        "report": manifest.cv_report_sha256,
    }:
        raise HybridRouterError("router bundle content hash mismatch")

    try:
        pipeline = joblib.load(io.BytesIO(model_bytes))
        _validate_fitted_pipeline(
            pipeline, expected_random_seed=manifest.random_seed
        )
        predictions = tuple(
            HybridRoutePrediction.model_validate_json(line)
            for line in oof_bytes.splitlines()
            if line.strip()
        )
        report = json.loads(report_bytes)
        metrics = HybridRouteMetrics.model_validate(report["metrics"])
    except Exception as error:
        raise HybridRouterError("router bundle payload is invalid") from error
    if tuple(sorted(predictions, key=lambda item: item.query_id)) != predictions:
        raise HybridRouterError("published OOF rows are not query-sorted")
    if evaluate_hybrid_predictions(predictions) != metrics:
        raise HybridRouterError("published CV report disagrees with OOF rows")
    if report.get("feature_contract_sha256") != FEATURE_CONTRACT_SHA256 or report.get("training_contract_sha256") != TRAINING_CONTRACT_SHA256:
        raise HybridRouterError("published CV report contract drifted")
    if report.get("examples_sha256") != manifest.oof_examples_sha256 or report.get("fold_mapping_sha256") != manifest.oof_fold_mapping_sha256:
        raise HybridRouterError("published CV report evidence hash drifted")
    if (
        report.get("schema_version") != 1
        or report.get("authorization_status") != AUTHORIZATION_STATUS
        or report.get("random_seed") != manifest.random_seed
        or tuple(report.get("fold_ids", ())) != tuple(
            sorted({item.fold_id for item in predictions})
        )
        or report.get("feature_contract") != FEATURE_CONTRACT
        or report.get("training_contract") != TRAINING_CONTRACT
    ):
        raise HybridRouterError("published CV report semantics drifted")
    return LoadedHybridRouterBundle(
        pipeline=pipeline,
        manifest=manifest,
        predictions=predictions,
        metrics=metrics,
    )


__all__ = [
    "AUTHORIZATION_STATUS",
    "CAPABILITY_ORDER",
    "DEFAULT_RANDOM_SEED",
    "FEATURE_CONTRACT",
    "FEATURE_CONTRACT_SHA256",
    "FOLD_COUNT",
    "FOLD_IDS",
    "HybridOOFResult",
    "HybridRouteExample",
    "HybridRouteDecision",
    "HybridRouteMetrics",
    "HybridRoutePrediction",
    "HybridRouterError",
    "HybridRouterManifest",
    "HybridThresholdPolicy",
    "HybridThresholdSelection",
    "LoadedHybridRouterBundle",
    "TRAINING_CONTRACT",
    "TRAINING_CONTRACT_SHA256",
    "build_hybrid_router",
    "decide_hybrid_route",
    "evaluate_hybrid_predictions",
    "fit_hybrid_router",
    "load_hybrid_router_bundle",
    "local_takeover",
    "make_hybrid_route_example",
    "project_user_turns",
    "publish_hybrid_router_bundle",
    "run_grouped_oof",
    "select_threshold_policy",
]
