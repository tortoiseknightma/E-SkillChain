"""Governance locks for Portfolio S1 Qwen visual Feedback.

Historical Qwen3.7 authorization and launch builders remain available for
terminal artifact verification.  Their parent Core authorization establishes
only catalog/image membership; its ``dashscope-qwen-assistant`` processor never
authorizes a Feedback role.

The Qwen3.7 declarations and loaders are historical and byte-compatible.  New
Feedback work uses the forward-only Qwen3.8-Max source/pricing identity.  The
historical v4/v11 locks bind the owner's exact Discovery240 plus one global
same-entry ``invalid_feedback_json`` retry under CNY94.  The v5/v12 identity
expanded the output envelope and retry plan under a CNY113 technical stop.
The forward source-v3/pricing-v6/role-v13 triad uniquely authorizes wholly
fresh Round3 JSON-object Feedback under a CNY135 cumulative technical stop.
The source-v4/pricing-v7/role-v14 triad is a separate forward-only identity for
strict JSON-Schema Round3 Feedback after the terminated JSON-object canary.  It
binds the terminated-canary cost into prior actual spend.  Live authority is
limited to canary12 plus three global retries under a fresh CNY10 stage cap;
the full243/CNY137 envelope is non-live planning and prior CNY150 authority is
not inherited.  Phase60 requires a new owner approval.  The forward
source-v5/pricing-v8/role-v15 triad records the completed strict-schema
canary as an immutable 12-result/15-call prefix and freezes a *pending*,
zero-call phase60 envelope.  It deliberately grants no live authority: the
fresh CNY28 budget stop and twelve new retry tokens require a separate owner
decision before an authorized successor lock can exist.

No function in this module calls a model provider.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
from typing import Annotated, Literal, Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    StringConstraints,
    field_validator,
    model_validator,
)

from skillchain.data.portfolio_remote_processing import (
    VerifiedPortfolioRemoteProcessingRuntime,
    require_verified_portfolio_remote_processing_runtime,
)
from skillchain.synthesis.store import (
    atomic_create_file,
    canonical_json_bytes,
    sha256_bytes,
)
from skillchain.tools.serialization import read_stable_regular_file


Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

QWEN37_FEEDBACK_PROVIDER = "qwen"
QWEN37_FEEDBACK_MODEL = "qwen3.7-plus-2026-05-26"
QWEN37_FEEDBACK_PROCESSOR = "dashscope-qwen37-feedback"
QWEN37_FEEDBACK_ENDPOINT_CONFIGURATION = "DASHSCOPE_BASE_URL"
QWEN37_FEEDBACK_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN37_FEEDBACK_CACHE_NAMESPACE = "feedback-evaluator-v9"
QWEN37_FEEDBACK_AUTHORIZATION_POLICY_VERSION = (
    "portfolio-s1-feedback-selected-assets-authorization-v3"
)
QWEN37_FEEDBACK_SOURCE_LOCK_POLICY_VERSION = (
    "portfolio-s1-qwen37-feedback-model-source-lock-v1"
)
QWEN37_FEEDBACK_PRICING_LOCK_POLICY_VERSION = (
    "portfolio-s1-qwen37-feedback-pricing-lock-v1"
)
QWEN37_FEEDBACK_PRICING_LOCK_POLICY_VERSION_V2 = (
    "portfolio-s1-qwen37-feedback-pricing-lock-v2"
)
QWEN37_FEEDBACK_LAUNCH_POLICY_VERSION = "portfolio-s1-qwen37-feedback-launch-lock-v1"
QWEN37_FEEDBACK_SELECTED_COUNT = 48
QWEN37_FEEDBACK_PROVIDER_CALL_CEILING = 48
QWEN37_FEEDBACK_THINKING_BUDGET = 2048
QWEN37_FEEDBACK_MAX_COMPLETION_TOKENS = 4096
QWEN37_FEEDBACK_MAX_COMPLETION_TOKENS_TOLERANCE = 10
QWEN37_FEEDBACK_TIMEOUT_SECONDS = 600
QWEN37_FEEDBACK_CONTEXT_WINDOW_TOKENS = 1_000_000
QWEN37_FEEDBACK_PROVIDER_MAX_OUTPUT_TOKENS = 131_072
QWEN37_FEEDBACK_PRICING_TIER_MAX_INPUT_TOKENS = 256_000
QWEN37_FEEDBACK_INPUT_CNY_PER_MILLION_TOKENS = 2
QWEN37_FEEDBACK_OUTPUT_CNY_PER_MILLION_TOKENS = 8
QWEN37_FEEDBACK_INPUT_RESERVATION_TOKENS = 20_000
QWEN37_FEEDBACK_OUTPUT_RESERVATION_TOKENS = 4_106
QWEN37_FEEDBACK_PER_CALL_RESERVATION_CNY = "0.072848000000"
QWEN37_FEEDBACK_MAXIMUM_RESERVATION_CNY = "3.496704000000"
QWEN37_FEEDBACK_PHASE_HARD_CAP_CNY = "4.000000000000"
QWEN37_FEEDBACK_SELECTED_COUNT_V2 = 240
QWEN37_FEEDBACK_PROVIDER_CALL_CEILING_V2 = 240
QWEN37_FEEDBACK_MAXIMUM_RESERVATION_CNY_V2 = "17.483520000000"
QWEN37_FEEDBACK_PHASE_HARD_CAP_CNY_V2 = "18.000000000000"
QWEN37_FEEDBACK_JSON_SCHEMA_POLICY_VERSION = "visual-feedback-output-json-schema-v1"
QWEN37_FEEDBACK_JSON_SCHEMA_NAME = "visual_feedback_output_v1"
# Updated only when the provider-facing schema itself is intentionally revised.
QWEN37_FEEDBACK_JSON_SCHEMA_SHA256 = (
    "af673bc72a52788b4a3871b030e95123e070d23cc51388e4abb06d0d5fa9667e"
)
QWEN37_FEEDBACK_TRANSPORT_POLICY_VERSION = (
    "visual-feedback-qwen-dashscope-json-schema-v5"
)
QWEN37_FEEDBACK_TRANSPORT_POLICY_SHA256 = (
    "092f36be5eb08e4fd58edc888fe273bee2b4295059048dbd9715d472e534f20c"
)
QWEN37_FEEDBACK_SOURCE_LOCK_FILE_SHA256 = (
    "8621e1823d0828efcd3cad1aba09f390dbbefa01fca509c525309e112b719d69"
)
QWEN37_FEEDBACK_SOURCE_LOCK_SHA256 = (
    "107b484f63cb9f7d05be56bd8f5f1182437b3f125fd2911e130fed9ebad2da07"
)
QWEN37_FEEDBACK_PRICING_LOCK_FILE_SHA256 = (
    "d6c8247b18d404a0c75102b1bed38ec59671b63c0a157f48d82fb8595e038015"
)
QWEN37_FEEDBACK_PRICING_LOCK_SHA256 = (
    "e68d1a0cde879174b06cc4230da848e595cd8ac7a7be8b4a8e383831cb4a8b30"
)
QWEN37_FEEDBACK_PRICING_LOCK_FILE_SHA256_V2 = (
    "8f2077cc23578ddfb56c7d7e06c2467632ec4ac61de17c46f279e88f25dcb718"
)
QWEN37_FEEDBACK_PRICING_LOCK_SHA256_V2 = (
    "5fd9c5a3770d1ac8dbf3d9c310e3bce4d88e35e8a04e3f54d10afb70f33af6e1"
)
QWEN37_FEEDBACK_ROLE_SELECTION_FILE_SHA256 = (
    "31ebdbe0af7d844ce3bf88bb0ab17d0c05640082618a4c24e1f022e38868efe8"
)
QWEN37_FEEDBACK_ROLE_SELECTION_SHA256 = (
    "9120e8abe4f667127c8ffc94f4175fc118188b2749ea7405340a03c44d7c369c"
)
QWEN37_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V2 = (
    "5289e3a233247ddfff339456fd3dab9e1d550a6fd37347ad61001bdb1ef134e1"
)
QWEN37_FEEDBACK_ROLE_SELECTION_SHA256_V2 = (
    "eef0214dffa79ad51c1d8e146789ccf756a43dd8293d473d198de30bb75b4f91"
)

# Active, forward-only Qwen3.8-Max Feedback identity.  Qwen3.7 constants above
# remain unchanged because historical artifacts bind their exact bytes.
QWEN38_FEEDBACK_PROVIDER = "qwen"
QWEN38_FEEDBACK_MODEL = "qwen3.8-max"
QWEN38_FEEDBACK_PROCESSOR = "dashscope-qwen38-feedback"
QWEN38_FEEDBACK_ENDPOINT_CONFIGURATION = "DASHSCOPE_BASE_URL"
QWEN38_FEEDBACK_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN38_FEEDBACK_CACHE_NAMESPACE = "feedback-evaluator-v11"
QWEN38_FEEDBACK_CACHE_NAMESPACE_V2 = "feedback-evaluator-v12"
QWEN38_FEEDBACK_CACHE_NAMESPACE_V3 = "feedback-evaluator-v13"
QWEN38_FEEDBACK_CACHE_NAMESPACE_V4 = "feedback-evaluator-v14"
QWEN38_FEEDBACK_RESULT_SCHEMA_VERSION_V2 = 6
QWEN38_FEEDBACK_RESULT_SCHEMA_VERSION_V3 = 7
QWEN38_FEEDBACK_RESULT_SCHEMA_VERSION_V4 = 8
QWEN38_FEEDBACK_BOUND_ARTIFACT_SCHEMA_VERSION_V2 = 5
QWEN38_FEEDBACK_BOUND_ARTIFACT_POLICY_VERSION_V2 = "portfolio-s1-bound-feedback-v5"
QWEN38_FEEDBACK_BOUND_ARTIFACT_SCHEMA_VERSION_V3 = 1
QWEN38_FEEDBACK_BOUND_ARTIFACT_POLICY_VERSION_V3 = (
    "portfolio-s1-bound-feedback-round3-v1"
)
QWEN38_FEEDBACK_BOUND_ARTIFACT_SCHEMA_VERSION_V4 = 2
QWEN38_FEEDBACK_BOUND_ARTIFACT_POLICY_VERSION_V4 = (
    "portfolio-s1-bound-feedback-round3-schema-v1"
)
QWEN38_FEEDBACK_SOURCE_LOCK_POLICY_VERSION = (
    "portfolio-s1-qwen38-feedback-model-source-lock-v1"
)
QWEN38_FEEDBACK_SOURCE_LOCK_POLICY_VERSION_V2 = (
    "portfolio-s1-qwen38-feedback-model-source-lock-v2"
)
QWEN38_FEEDBACK_SOURCE_LOCK_POLICY_VERSION_V3 = (
    "portfolio-s1-qwen38-feedback-model-source-lock-v3"
)
QWEN38_FEEDBACK_SOURCE_LOCK_POLICY_VERSION_V4 = (
    "portfolio-s1-qwen38-feedback-model-source-lock-v4"
)
QWEN38_FEEDBACK_SOURCE_LOCK_POLICY_VERSION_V5 = (
    "portfolio-s1-qwen38-feedback-model-source-lock-v5"
)
QWEN38_FEEDBACK_PRICING_LOCK_POLICY_VERSION_V3 = (
    "portfolio-s1-qwen38-feedback-pricing-lock-v3"
)
QWEN38_FEEDBACK_PRICING_LOCK_POLICY_VERSION_V4 = (
    "portfolio-s1-qwen38-feedback-pricing-lock-v4"
)
QWEN38_FEEDBACK_PRICING_LOCK_POLICY_VERSION_V5 = (
    "portfolio-s1-qwen38-feedback-pricing-lock-v5"
)
QWEN38_FEEDBACK_PRICING_LOCK_POLICY_VERSION_V6 = (
    "portfolio-s1-qwen38-feedback-pricing-lock-v6"
)
QWEN38_FEEDBACK_PRICING_LOCK_POLICY_VERSION_V7 = (
    "portfolio-s1-qwen38-feedback-pricing-lock-v7"
)
QWEN38_FEEDBACK_PRICING_LOCK_POLICY_VERSION_V8 = (
    "portfolio-s1-qwen38-feedback-pricing-lock-v8"
)
QWEN38_FEEDBACK_SELECTED_COUNT = 240
QWEN38_FEEDBACK_PROVIDER_CALL_CEILING = 240
QWEN38_FEEDBACK_PROVIDER_CALL_CEILING_V2 = 241
QWEN38_FEEDBACK_PROVIDER_CALL_CEILING_V3 = 243
QWEN38_FEEDBACK_THINKING_BUDGET = 2048
QWEN38_FEEDBACK_MAX_COMPLETION_TOKENS = 4096
QWEN38_FEEDBACK_MAX_COMPLETION_TOKENS_V2 = 6144
QWEN38_FEEDBACK_MAX_COMPLETION_TOKENS_TOLERANCE = 10
QWEN38_FEEDBACK_TIMEOUT_SECONDS = 600
QWEN38_FEEDBACK_PRICING_TIER_MAX_INPUT_TOKENS = 1_000_000
QWEN38_FEEDBACK_INPUT_CNY_PER_MILLION_TOKENS = 12
QWEN38_FEEDBACK_OUTPUT_CNY_PER_MILLION_TOKENS = 36
QWEN38_FEEDBACK_INPUT_RESERVATION_TOKENS = 20_000
QWEN38_FEEDBACK_OUTPUT_RESERVATION_TOKENS = 4_106
QWEN38_FEEDBACK_OUTPUT_RESERVATION_TOKENS_V2 = 6_154
QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY = "0.387816000000"
QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY_V2 = "0.461544000000"
QWEN38_FEEDBACK_MAXIMUM_RESERVATION_CNY = "93.075840000000"
QWEN38_FEEDBACK_MAXIMUM_RESERVATION_CNY_V2 = "93.463656000000"
QWEN38_FEEDBACK_MAXIMUM_RESERVATION_CNY_V3 = "112.155192000000"
QWEN38_FEEDBACK_TECHNICAL_PHASE_HARD_CAP_CNY = "94.000000000000"
QWEN38_FEEDBACK_TECHNICAL_PHASE_HARD_CAP_CNY_V2 = "113.000000000000"
QWEN38_FEEDBACK_OWNER_AUTHORIZED_BUDGET_CEILING_CNY = "150.000000000000"
QWEN38_FEEDBACK_ROUND3_PRIOR_ACTUAL_COST_CNY = "22.764432000000"
QWEN38_FEEDBACK_ROUND3_CUMULATIVE_MAXIMUM_CNY = "134.919624000000"
QWEN38_FEEDBACK_ROUND3_CUMULATIVE_TECHNICAL_HARD_CAP_CNY = "135.000000000000"
QWEN38_FEEDBACK_ROUND3_SCHEMA_PRIOR_ACTUAL_COST_CNY = "24.319500000000"
# This full-run envelope is planning-only until the owner separately approves
# phase60 or beyond.  It is not live authority for the current wire change.
QWEN38_FEEDBACK_ROUND3_SCHEMA_CUMULATIVE_MAXIMUM_CNY = "136.474692000000"
QWEN38_FEEDBACK_ROUND3_SCHEMA_CUMULATIVE_TECHNICAL_HARD_CAP_CNY = "137.000000000000"
QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_SELECTED_COUNT = 12
QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_PROVIDER_CALL_CEILING = 15
QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_FRESH_RESERVATION_CNY = "6.923160000000"
QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_FRESH_STAGE_HARD_CAP_CNY = "10.000000000000"
QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_CUMULATIVE_MAXIMUM_CNY = "31.242660000000"
QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_CUMULATIVE_HARD_CAP_CNY = "34.319500000000"
QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_ACTUAL_COST_CNY = "1.944600000000"
QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_CUMULATIVE_ACTUAL_CNY = "26.264100000000"
QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_RUN_FILE_SHA256 = (
    "46bc2bb477acae1d367f2e4200b41ad8c2194aa0c5044667a414fb47d371d40b"
)
QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_RUN_SHA256 = (
    "2d40e9ee4d9cc0d3d92585f1126ddcb1362ee7ef32c35f60c86593d668c0cd72"
)
QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_ARTIFACT_SET_SHA256 = (
    "98b1185b163230998bbc760600ff62f44561d99346f3dd6e66d0c3f8d4730503"
)
QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_SELECTION_SHA256 = (
    "ce55b4d47d57a8313aabf014947c85a05150d3a6a4d3338d53548f4a4b983674"
)
QWEN38_FEEDBACK_PHASE60_PREFIX_SELECTED_COUNT = 12
QWEN38_FEEDBACK_PHASE60_PREFIX_PROVIDER_CALL_COUNT = 15
QWEN38_FEEDBACK_PHASE60_PREFIX_RETRY_COUNT = 3
QWEN38_FEEDBACK_PHASE60_NEW_FIRST_CALL_COUNT = 48
QWEN38_FEEDBACK_PHASE60_NEW_RETRY_TOKEN_COUNT = 12
QWEN38_FEEDBACK_PHASE60_NEW_PROVIDER_CALL_CEILING = 60
QWEN38_FEEDBACK_PHASE60_CUMULATIVE_PROVIDER_CALL_CEILING = 75
QWEN38_FEEDBACK_PHASE60_FRESH_MAXIMUM_RESERVATION_CNY = "27.692640000000"
QWEN38_FEEDBACK_PHASE60_FRESH_TECHNICAL_HARD_CAP_CNY = "28.000000000000"
QWEN38_FEEDBACK_PHASE60_CUMULATIVE_MAXIMUM_CNY = "53.956740000000"
QWEN38_FEEDBACK_PHASE60_CUMULATIVE_TECHNICAL_HARD_CAP_CNY = "54.264100000000"
QWEN38_FEEDBACK_JSON_SCHEMA_POLICY_VERSION = QWEN37_FEEDBACK_JSON_SCHEMA_POLICY_VERSION
QWEN38_FEEDBACK_JSON_SCHEMA_NAME = QWEN37_FEEDBACK_JSON_SCHEMA_NAME
QWEN38_FEEDBACK_JSON_SCHEMA_SHA256 = QWEN37_FEEDBACK_JSON_SCHEMA_SHA256
QWEN38_FEEDBACK_TRANSPORT_POLICY_VERSION = (
    "visual-feedback-qwen38-dashscope-json-schema-v6"
)
QWEN38_FEEDBACK_TRANSPORT_POLICY_SHA256 = (
    "71b8608529394852b455a11099741e6dc5f5b04fef875998e3652c81f351aa29"
)
QWEN38_FEEDBACK_TRANSPORT_POLICY_VERSION_V7 = (
    "visual-feedback-qwen38-dashscope-json-schema-v7"
)
QWEN38_FEEDBACK_TRANSPORT_POLICY_SHA256_V7 = (
    "462b2ef6f7afb0d618f0f29f0d24590aaac45a00ece52cd99643151045a36a0b"
)
ROUND3_PRIMARY_TRANSPORT_POLICY_VERSION_V1 = (
    "visual-feedback-qwen38-dashscope-json-object-round3-primary-v1"
)
ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1 = (
    "7f9e91479e024ac1ade536d57f30ab1f7bd8f7efdd9865e21cc9fd2776769313"
)
QWEN38_FEEDBACK_ROUND3_WIRE_KIND_V1 = "round3_primary_json_object_v1"
ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_VERSION_V1 = (
    "visual-feedback-qwen38-dashscope-json-schema-round3-primary-v1"
)
ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1 = (
    "e057e69ec8f40185a909163e3c54ad4c0132e4ae2df3537e662c1e2cb3f63845"
)
QWEN38_FEEDBACK_ROUND3_WIRE_KIND_V2 = "round3_primary_json_schema_v1"
QWEN38_FEEDBACK_ROUND3_PROMPT_POLICY_VERSION_V6 = (
    "visual-feedback-gcs-policy-labels-prompt-v6"
)
QWEN38_FEEDBACK_ROUND3_PROMPT_POLICY_SHA256_V6 = (
    "c4c6a0afcc472de09a1c8c27c1eaa276590b60ecfceb348b3b634a56f24a2f3f"
)
QWEN38_FEEDBACK_ROUND3_PARSER_POLICY_VERSION_V3 = "visual-feedback-free-text-trim-v3"
QWEN38_FEEDBACK_ROUND3_PARSER_POLICY_SHA256_V3 = (
    "d2858a1aa2db6efc524c6f826817b0787e7506a67273066fbf2e891cbb1c5016"
)
QWEN38_FEEDBACK_ROUND3_RETRY_POLICY_VERSION_V1 = (
    "portfolio-s1-feedback-round3-global-retry-v1"
)
QWEN38_FEEDBACK_ROUND3_RETRY_POLICY_SHA256_V1 = (
    "98e931d2d5c1031e50769b13364446f72d8f4b94d3036a4acc26cea652ca94f3"
)
QWEN38_FEEDBACK_ROUND3_RETRY_POLICY_VERSION_V2 = (
    "portfolio-s1-feedback-round3-global-retry-v2"
)
QWEN38_FEEDBACK_ROUND3_RETRY_POLICY_SHA256_V2 = (
    "f19c1d78e5d6604a9fa8d54bc0e13493d23b05f21b66af89353c9691dd92d31b"
)
ROUND3_PHASE60_RETRY_POLICY_VERSION_V1 = (
    "portfolio-s1-feedback-round3-phase60-global-retry-v1"
)
QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_VERSION_V1 = (
    "portfolio-s1-feedback-global-schema-retry-v1"
)
QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V1 = (
    "6ba3799fce29820466446c6ec0ee98312c6e889fc17c255b25453b9f70694995"
)
QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_VERSION_V2 = (
    "portfolio-s1-feedback-global-schema-or-length-retry-v2"
)
QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V2 = (
    "f6ca1511748f0965831f5fda3a9df0e1c7b141f322b21ca82aba8a443837071a"
)
QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256 = (
    "657d14320c9742dfb9d1eecde92cce4c932e0fb5dea12452437fafc5e1c6ecae"
)
QWEN38_FEEDBACK_SOURCE_LOCK_SHA256 = (
    "0f8150c924ab82960c8feda85d82ec4257b7e2889995e56ec22f13aa2611f822"
)
QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V2 = (
    "c69af8b080ceccffbe9303844fe55245424b173201d44357540c1685d4843abb"
)
QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V2 = (
    "3d4e3b8511b00a75a88efc2c509039108d987eb7f960cf7941e9798626783901"
)
QWEN38_FEEDBACK_SOURCE_LOCK_RELATIVE_PATH_V3 = (
    "specs/authoring/qwen3.8-max-feedback-source-lock-v3.json"
)
QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V3 = (
    "eb075d4d09741a06e77050f79539bc85610323af21b160e36ee502be96c7c1a3"
)
QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V3 = (
    "5373c79d22a81c52598f01e80aa60683a8e49c196472838f25391aa83aa2d91b"
)
QWEN38_FEEDBACK_SOURCE_LOCK_RELATIVE_PATH_V4 = (
    "specs/authoring/qwen3.8-max-feedback-source-lock-v4.json"
)
QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V4 = (
    "270ad7bc8f673aa1785100a39da546bcea00461e4cc8161a1cbe184a9e8d2d7c"
)
QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V4 = (
    "68e4f1fe7c48cb2f89bb9b58d8d2fb00f2ad088d3ef51f3139c09575c06d62b9"
)
QWEN38_FEEDBACK_SOURCE_LOCK_RELATIVE_PATH_V5 = (
    "specs/authoring/qwen3.8-max-feedback-source-lock-v5.json"
)
QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V5 = (
    "7bca1ac1d1326c7150f033408497d00214e3de2223e476bdf3887f2f3853a304"
)
QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V5 = (
    "9ae82da8d6096ee108ad8339044d7d29ccd22644f10b123c1697b71106c5e046"
)
QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V3 = (
    "bd04d13702052f553ec643818ea7b0cf752d90dce700b2cfcd7996f17e4dfdf8"
)
QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V3 = (
    "98002982e8a8815914d021480aac57187e2d81a59556731a72612daeccf53082"
)
QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V4 = (
    "2654264a6897b8e1f071ebdab00479f1d4f70344b776aa1fd8d3ae685095ca1c"
)
QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V4 = (
    "722c75e313e3843d67c5f889b1d7c2812339b3f6feac44fcd7407d21a675ccf7"
)
QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V5 = (
    "9cfaa1fac362c2b1d2a46dcd55afad8157abb8ab8eeb341904330b094dda0d9a"
)
QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V5 = (
    "15f2582f26e6b5c720e261f42e0b536ec7bbb4dd262f0347d8445c99c42e07bf"
)
QWEN38_FEEDBACK_PRICING_LOCK_RELATIVE_PATH_V6 = (
    "specs/authoring/price-qwen3.8-max-feedback-v6.json"
)
QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V6 = (
    "e23cc28625cde23accc3f88d456acfa20a5ba4b2f351301a04524ba7e418487c"
)
QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V6 = (
    "c51d64408d37fb5ca2ef44065e6c500b46bf23b8d41a31c130a59e171e318af9"
)
QWEN38_FEEDBACK_PRICING_LOCK_RELATIVE_PATH_V7 = (
    "specs/authoring/price-qwen3.8-max-feedback-v7.json"
)
QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V7 = (
    "7ace94ece2e441cb4fca8cddf0d171fa30a0d8f02874fa88b23ce961c36ca8ea"
)
QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V7 = (
    "b295abe255e5e76945ef38df4305bacd7a8983d36bf2d360ca53a72e6de72c89"
)
QWEN38_FEEDBACK_PRICING_LOCK_RELATIVE_PATH_V8 = (
    "specs/authoring/price-qwen3.8-max-feedback-v8.json"
)
QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V8 = (
    "8c7dafe5be2ed046b2452e8392fefb31eedce215d28a70e24dd75f817a46736f"
)
QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V8 = (
    "7bb49d2533385db1d4edbfd1396ffdef8503599199c4da67048882f6d815dd12"
)
QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256 = (
    "39e8d099f0b303725ee0be59f758560627ecbc3ce156524da4f2a2d3d20f3501"
)
QWEN38_FEEDBACK_ROLE_SELECTION_SHA256 = (
    "be5f7f7ee1f30a81d7d2a8423384adb41d74e4673000901e883af704e9eead16"
)
QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V11 = (
    "048362dd770c626ac4c2293b1d8a28cb4fcb2a3a240c4f79dca745e4dbb5c50c"
)
QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V11 = (
    "8f9252b9dc4ddf6f3d6569f783e150f55f7f6cd422b77126e73b5c38ba99e432"
)
QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V12 = (
    "474826b4fdd89af7df3f520e5bc27af880291b2b76239a4abca7fd40679f1cef"
)
QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V12 = (
    "2ada8b9f973174a26abd57e71b253986add75e6dbe2316aef603cf1fa49d5db9"
)
QWEN38_FEEDBACK_ROLE_SELECTION_RELATIVE_PATH_V13 = (
    "specs/authoring/model-role-selection-v13.json"
)
QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V13 = (
    "f8444da36fc65dbd63784b4668fda3c39ce79e414dbb9a4c5b053504d52fca40"
)
QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V13 = (
    "231ddf2bcb37ef489749822412132356184a4eb167cb335424f64d1f45e074f2"
)
QWEN38_FEEDBACK_ROLE_SELECTION_RELATIVE_PATH_V14 = (
    "specs/authoring/model-role-selection-v14.json"
)
QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V14 = (
    "999cfda04396323ac24ff49dd42f8494bde99794dec3a088f1a60d6787212a41"
)
QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V14 = (
    "d126ee787952486e796f122676dc9eb641cdf200caf39bfe3b78568c210079a9"
)
QWEN38_FEEDBACK_ROLE_SELECTION_RELATIVE_PATH_V15 = (
    "specs/authoring/model-role-selection-v15.json"
)
QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V15 = (
    "6c4425883faede04589ad9ce0da06ebe46d69c5a329bb957429571be634f8e24"
)
QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V15 = (
    "cda5798fa167e2c89cc3af95a78ebee33437ce87e2fa83ed63d7e8b446de7f68"
)

_AUTHORIZATION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class PortfolioS1QwenFeedbackGovernanceError(ValueError):
    """A Qwen Feedback governance artifact is invalid."""


class _StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def _self_hash(model: BaseModel, field: str) -> str:
    return sha256_bytes(
        canonical_json_bytes(model.model_dump(mode="json", exclude={field}))
    )


def _canonical_text(value: str, label: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{label} must be non-blank and trimmed")
    return value


def round3_phase60_retry_policy_v1() -> dict[str, object]:
    """Return the owner-approved live retry policy for the phase60 overlay."""

    return {
        "policy_version": ROUND3_PHASE60_RETRY_POLICY_VERSION_V1,
        "scope": "schema-canary-prefix12-plus-fresh-selection-ordinals-13-through-60",
        "prefix_selected_count": QWEN38_FEEDBACK_PHASE60_PREFIX_SELECTED_COUNT,
        "prefix_provider_call_count": (
            QWEN38_FEEDBACK_PHASE60_PREFIX_PROVIDER_CALL_COUNT
        ),
        "prefix_retry_claims_consumed": QWEN38_FEEDBACK_PHASE60_PREFIX_RETRY_COUNT,
        "prefix_retry_claims_reusable": False,
        "new_first_calls": QWEN38_FEEDBACK_PHASE60_NEW_FIRST_CALL_COUNT,
        "new_global_retry_ceiling": QWEN38_FEEDBACK_PHASE60_NEW_RETRY_TOKEN_COUNT,
        "new_provider_call_ceiling": (
            QWEN38_FEEDBACK_PHASE60_NEW_PROVIDER_CALL_CEILING
        ),
        "cumulative_provider_call_ceiling": (
            QWEN38_FEEDBACK_PHASE60_CUMULATIVE_PROVIDER_CALL_CEILING
        ),
        "max_lifetime_attempts_per_new_entry": 2,
        "provider_internal_max_attempts": 1,
        "retry_eligibility": [
            "strict-parser-failure-without-policy-label-only-failure",
            "length-parser-failure",
            "reasoning-present-empty-assistant-content",
        ],
        "terminal_without_retry": [
            "provider-error",
            "timeout",
            "refusal",
            "tool-call",
            "input-echo",
            "privacy-failure",
            "orphan",
            "thirteenth-new-eligible-failure",
        ],
        "live_provider_calls_authorized": True,
        "owner_phase60_budget_and_retry_approval_status": "granted",
        "owner_approval_decision_source": "current_user_instruction",
        "owner_approved_on": "2026-08-11",
        "owner_approved_fresh_maximum_reservation_cny": (
            QWEN38_FEEDBACK_PHASE60_FRESH_MAXIMUM_RESERVATION_CNY
        ),
        "owner_approved_fresh_technical_hard_cap_cny": (
            QWEN38_FEEDBACK_PHASE60_FRESH_TECHNICAL_HARD_CAP_CNY
        ),
        "owner_approved_cumulative_technical_hard_cap_cny": (
            QWEN38_FEEDBACK_PHASE60_CUMULATIVE_TECHNICAL_HARD_CAP_CNY
        ),
        "phase120_requires_new_owner_approval": True,
    }


ROUND3_PHASE60_RETRY_POLICY_SHA256_V1 = sha256_bytes(
    canonical_json_bytes(round3_phase60_retry_policy_v1())
)


class _SelectionEntry(Protocol):
    query_id: str
    asset_id: str
    image_sha256: str


class _Selection(Protocol):
    selection_sha256: str
    entries: tuple[_SelectionEntry, ...]


class Qwen37FeedbackSourceEvidenceV1(_StrictFrozenModel):
    source_id: Literal[
        "model-capabilities",
        "structured-output",
        "deep-thinking",
        "chat-completions",
        "pricing",
    ]
    url: str
    retrieved_on: Literal["2026-08-10"] = "2026-08-10"
    facts: tuple[str, ...]

    @field_validator("facts", mode="before")
    @classmethod
    def _facts_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("url")
    @classmethod
    def _url(cls, value: str) -> str:
        value = _canonical_text(value, "source URL")
        if not value.startswith("https://help.aliyun.com/"):
            raise ValueError("Qwen source evidence must use official Alibaba docs")
        return value

    @model_validator(mode="after")
    def _validate_facts(self) -> Self:
        if not self.facts or any(
            not item or item != item.strip() for item in self.facts
        ):
            raise ValueError("source evidence facts must be non-empty and canonical")
        return self


class Qwen37FeedbackModelSourceLockV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-qwen37-feedback-model-source-lock"] = (
        "portfolio-s1-qwen37-feedback-model-source-lock"
    )
    policy_version: Literal["portfolio-s1-qwen37-feedback-model-source-lock-v1"] = (
        QWEN37_FEEDBACK_SOURCE_LOCK_POLICY_VERSION
    )
    provider: Literal["qwen"] = "qwen"
    model: Literal["qwen3.7-plus-2026-05-26"] = QWEN37_FEEDBACK_MODEL
    deployment_region: Literal["china-beijing"] = "china-beijing"
    endpoint_configuration: Literal["DASHSCOPE_BASE_URL"] = (
        QWEN37_FEEDBACK_ENDPOINT_CONFIGURATION
    )
    endpoint: Literal["https://dashscope.aliyuncs.com/compatible-mode/v1"] = (
        QWEN37_FEEDBACK_ENDPOINT
    )
    input_modalities: tuple[Literal["text", "image", "video"], ...] = (
        "text",
        "image",
        "video",
    )
    output_modalities: tuple[Literal["text"], ...] = ("text",)
    context_window_tokens: Literal[1000000] = QWEN37_FEEDBACK_CONTEXT_WINDOW_TOKENS
    provider_max_output_tokens: Literal[131072] = (
        QWEN37_FEEDBACK_PROVIDER_MAX_OUTPUT_TOKENS
    )
    structured_output_supported: Literal[True] = True
    thinking_mode_supported: Literal[True] = True
    structured_output_requires_non_thinking: Literal[False] = False
    structured_output_with_thinking_supported: Literal[True] = True
    structured_output_json_schema_supported: Literal[True] = True
    structured_output_response_format: Literal["json_schema"] = "json_schema"
    structured_output_strict: Literal[True] = True
    structured_output_prompt_must_contain_json: Literal[False] = False
    structured_output_max_tokens_guidance: Literal[
        "official-docs-recommend-omitting-max_tokens-to-avoid-truncation"
    ] = "official-docs-recommend-omitting-max_tokens-to-avoid-truncation"
    max_tokens_excludes_reasoning: Literal[True] = True
    max_completion_tokens_includes_reasoning_and_answer: Literal[True] = True
    max_completion_tokens_documented_upper_tolerance_tokens: Literal[10] = (
        QWEN37_FEEDBACK_MAX_COMPLETION_TOKENS_TOLERANCE
    )
    visual_input_supported: Literal[True] = True
    frozen_snapshot: Literal[True] = True
    evidence: tuple[Qwen37FeedbackSourceEvidenceV1, ...]
    source_lock_sha256: Sha256

    @field_validator("input_modalities", "output_modalities", "evidence", mode="before")
    @classmethod
    def _tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_lock(self) -> Self:
        if tuple(item.source_id for item in self.evidence) != (
            "model-capabilities",
            "structured-output",
            "deep-thinking",
            "chat-completions",
            "pricing",
        ):
            raise ValueError("Qwen source evidence order or coverage differs")
        if self.source_lock_sha256 != _self_hash(self, "source_lock_sha256"):
            raise ValueError("Qwen model source lock self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class Qwen38FeedbackSourceEvidenceV1(_StrictFrozenModel):
    source_id: Literal[
        "model-catalog",
        "structured-output",
        "chat-completions",
        "pricing",
    ]
    url: str
    retrieved_on: Literal["2026-08-10"] = "2026-08-10"
    facts: tuple[str, ...]

    @field_validator("facts", mode="before")
    @classmethod
    def _facts_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator("url")
    @classmethod
    def _url(cls, value: str) -> str:
        value = _canonical_text(value, "source URL")
        if not value.startswith("https://help.aliyun.com/"):
            raise ValueError("Qwen source evidence must use official Alibaba docs")
        return value

    @model_validator(mode="after")
    def _validate_facts(self) -> Self:
        if not self.facts or any(
            not item or item != item.strip() for item in self.facts
        ):
            raise ValueError("source evidence facts must be non-empty and canonical")
        return self


class Qwen38FeedbackModelSourceLockV1(_StrictFrozenModel):
    """Official-source identity for the active, undated Qwen3.8-Max alias."""

    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-qwen38-feedback-model-source-lock"] = (
        "portfolio-s1-qwen38-feedback-model-source-lock"
    )
    policy_version: Literal["portfolio-s1-qwen38-feedback-model-source-lock-v1"] = (
        QWEN38_FEEDBACK_SOURCE_LOCK_POLICY_VERSION
    )
    provider: Literal["qwen"] = QWEN38_FEEDBACK_PROVIDER
    model: Literal["qwen3.8-max"] = QWEN38_FEEDBACK_MODEL
    model_identity_kind: Literal["moving_alias"] = "moving_alias"
    frozen_snapshot: Literal[False] = False
    deployment_region: Literal["china-beijing"] = "china-beijing"
    endpoint_configuration: Literal["DASHSCOPE_BASE_URL"] = (
        QWEN38_FEEDBACK_ENDPOINT_CONFIGURATION
    )
    endpoint: Literal["https://dashscope.aliyuncs.com/compatible-mode/v1"] = (
        QWEN38_FEEDBACK_ENDPOINT
    )
    input_modalities: tuple[Literal["text", "image"], ...] = ("text", "image")
    output_modalities: tuple[Literal["text"], ...] = ("text",)
    visual_input_supported: Literal[True] = True
    thinking_mode_supported: Literal[True] = True
    thinking_budget_supported: Literal[True] = True
    structured_output_supported: Literal[True] = True
    structured_output_json_schema_supported: Literal[True] = True
    structured_output_response_format: Literal["json_schema"] = "json_schema"
    structured_output_strict: Literal[True] = True
    max_completion_tokens_includes_reasoning_and_answer: Literal[True] = True
    max_completion_tokens_documented_upper_tolerance_tokens: Literal[10] = (
        QWEN38_FEEDBACK_MAX_COMPLETION_TOKENS_TOLERANCE
    )
    evidence: tuple[Qwen38FeedbackSourceEvidenceV1, ...]
    source_lock_sha256: Sha256

    @field_validator("input_modalities", "output_modalities", "evidence", mode="before")
    @classmethod
    def _tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_lock(self) -> Self:
        if tuple(item.source_id for item in self.evidence) != (
            "model-catalog",
            "structured-output",
            "chat-completions",
            "pricing",
        ):
            raise ValueError("Qwen3.8 source evidence order or coverage differs")
        if self.source_lock_sha256 != _self_hash(self, "source_lock_sha256"):
            raise ValueError("Qwen3.8 model source lock self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class Qwen38FeedbackModelSourceLockV2(Qwen38FeedbackModelSourceLockV1):
    """Forward wire lock for the fresh-v3 6,144-token Feedback run."""

    schema_version: Literal[2] = 2
    policy_version: Literal["portfolio-s1-qwen38-feedback-model-source-lock-v2"] = (
        QWEN38_FEEDBACK_SOURCE_LOCK_POLICY_VERSION_V2
    )
    enable_thinking: Literal[True] = True
    thinking_budget: Literal[2048] = QWEN38_FEEDBACK_THINKING_BUDGET
    requested_response_format: Literal["json_schema"] = "json_schema"
    requested_json_schema_policy_version: Literal[
        "visual-feedback-output-json-schema-v1"
    ] = QWEN38_FEEDBACK_JSON_SCHEMA_POLICY_VERSION
    requested_json_schema_name: Literal["visual_feedback_output_v1"] = (
        QWEN38_FEEDBACK_JSON_SCHEMA_NAME
    )
    requested_json_schema_sha256: Literal[QWEN38_FEEDBACK_JSON_SCHEMA_SHA256] = (
        QWEN38_FEEDBACK_JSON_SCHEMA_SHA256
    )
    requested_json_schema_strict: Literal[True] = True
    max_tokens: None = None
    max_completion_tokens: Literal[6144] = QWEN38_FEEDBACK_MAX_COMPLETION_TOKENS_V2
    output_token_reservation_ceiling_per_call: Literal[6154] = (
        QWEN38_FEEDBACK_OUTPUT_RESERVATION_TOKENS_V2
    )
    timeout_seconds: Literal[600] = QWEN38_FEEDBACK_TIMEOUT_SECONDS
    requested_stream: Literal[False] = False
    transport_policy_version: Literal[
        "visual-feedback-qwen38-dashscope-json-schema-v7"
    ] = QWEN38_FEEDBACK_TRANSPORT_POLICY_VERSION_V7
    transport_policy_sha256: Literal[QWEN38_FEEDBACK_TRANSPORT_POLICY_SHA256_V7] = (
        QWEN38_FEEDBACK_TRANSPORT_POLICY_SHA256_V7
    )
    cache_namespace: Literal["feedback-evaluator-v12"] = (
        QWEN38_FEEDBACK_CACHE_NAMESPACE_V2
    )
    result_schema_version: Literal[6] = QWEN38_FEEDBACK_RESULT_SCHEMA_VERSION_V2
    bound_artifact_schema_version: Literal[5] = (
        QWEN38_FEEDBACK_BOUND_ARTIFACT_SCHEMA_VERSION_V2
    )
    bound_artifact_policy_version: Literal["portfolio-s1-bound-feedback-v5"] = (
        QWEN38_FEEDBACK_BOUND_ARTIFACT_POLICY_VERSION_V2
    )


class Qwen38FeedbackModelSourceLockV3(Qwen38FeedbackModelSourceLockV1):
    """Forward-only source/wire lock for the wholly fresh Round3 run."""

    schema_version: Literal[3] = 3
    policy_version: Literal["portfolio-s1-qwen38-feedback-model-source-lock-v3"] = (
        QWEN38_FEEDBACK_SOURCE_LOCK_POLICY_VERSION_V3
    )
    structured_output_json_object_supported: Literal[True] = True
    structured_output_response_format: Literal["json_object"] = "json_object"
    structured_output_json_schema_used: Literal[False] = False
    structured_output_strict: Literal[False] = False
    enable_thinking: Literal[True] = True
    thinking_budget: Literal[2048] = QWEN38_FEEDBACK_THINKING_BUDGET
    requested_response_format: Literal["json_object"] = "json_object"
    requested_json_schema: Literal[False] = False
    requested_json_schema_strict: Literal[False] = False
    max_tokens: None = None
    max_completion_tokens: Literal[6144] = QWEN38_FEEDBACK_MAX_COMPLETION_TOKENS_V2
    output_token_reservation_ceiling_per_call: Literal[6154] = (
        QWEN38_FEEDBACK_OUTPUT_RESERVATION_TOKENS_V2
    )
    timeout_seconds: Literal[600] = QWEN38_FEEDBACK_TIMEOUT_SECONDS
    requested_stream: Literal[False] = False
    temperature: None = None
    top_p: None = None
    seed: None = None
    provider_internal_max_attempts: Literal[1] = 1
    wire_kind: Literal["round3_primary_json_object_v1"] = (
        QWEN38_FEEDBACK_ROUND3_WIRE_KIND_V1
    )
    transport_policy_version: Literal[
        "visual-feedback-qwen38-dashscope-json-object-round3-primary-v1"
    ] = ROUND3_PRIMARY_TRANSPORT_POLICY_VERSION_V1
    transport_policy_sha256: Literal[ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1] = (
        ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1
    )
    prompt_policy_version: Literal["visual-feedback-gcs-policy-labels-prompt-v6"] = (
        QWEN38_FEEDBACK_ROUND3_PROMPT_POLICY_VERSION_V6
    )
    prompt_policy_sha256: Literal[QWEN38_FEEDBACK_ROUND3_PROMPT_POLICY_SHA256_V6] = (
        QWEN38_FEEDBACK_ROUND3_PROMPT_POLICY_SHA256_V6
    )
    parser_policy_version: Literal["visual-feedback-free-text-trim-v3"] = (
        QWEN38_FEEDBACK_ROUND3_PARSER_POLICY_VERSION_V3
    )
    parser_policy_sha256: Literal[QWEN38_FEEDBACK_ROUND3_PARSER_POLICY_SHA256_V3] = (
        QWEN38_FEEDBACK_ROUND3_PARSER_POLICY_SHA256_V3
    )
    cache_namespace: Literal["feedback-evaluator-v13"] = (
        QWEN38_FEEDBACK_CACHE_NAMESPACE_V3
    )
    result_schema_version: Literal[7] = QWEN38_FEEDBACK_RESULT_SCHEMA_VERSION_V3
    bound_artifact_schema_version: Literal[1] = (
        QWEN38_FEEDBACK_BOUND_ARTIFACT_SCHEMA_VERSION_V3
    )
    bound_artifact_policy_version: Literal["portfolio-s1-bound-feedback-round3-v1"] = (
        QWEN38_FEEDBACK_BOUND_ARTIFACT_POLICY_VERSION_V3
    )
    fixed_selected_query_count: Literal[240] = QWEN38_FEEDBACK_SELECTED_COUNT
    provider_call_ceiling: Literal[243] = QWEN38_FEEDBACK_PROVIDER_CALL_CEILING_V3
    live_call_authority: Literal[True] = True
    image_source_membership_ancestry_only: Literal[True] = True
    old_remote_auth_does_not_authorize_round3_transport: Literal[True] = True
    round3_live_authority_requires_exact_v3_v6_v13_triad: Literal[True] = True

    @model_validator(mode="after")
    def _validate_v3_lock(self) -> Self:
        if (
            self.requested_response_format != "json_object"
            or self.structured_output_response_format != "json_object"
            or self.structured_output_json_schema_used
            or self.requested_json_schema
            or not self.old_remote_auth_does_not_authorize_round3_transport
        ):
            raise ValueError("Qwen3.8 Feedback Round3 source/wire lock drifted")
        if self.source_lock_sha256 != _self_hash(self, "source_lock_sha256"):
            raise ValueError("Qwen3.8 Feedback model source lock v3 self hash mismatch")
        return self


class Qwen38FeedbackModelSourceLockV4(Qwen38FeedbackModelSourceLockV1):
    """Strict JSON-Schema wire lock after the terminated Round3 canary."""

    schema_version: Literal[4] = 4
    policy_version: Literal["portfolio-s1-qwen38-feedback-model-source-lock-v4"] = (
        QWEN38_FEEDBACK_SOURCE_LOCK_POLICY_VERSION_V4
    )
    structured_output_json_object_supported: Literal[True] = True
    structured_output_response_format: Literal["json_schema"] = "json_schema"
    structured_output_json_schema_used: Literal[True] = True
    structured_output_strict: Literal[True] = True
    enable_thinking: Literal[True] = True
    thinking_budget: Literal[2048] = QWEN38_FEEDBACK_THINKING_BUDGET
    requested_response_format: Literal["json_schema"] = "json_schema"
    requested_json_schema: Literal[True] = True
    requested_json_schema_policy_version: Literal[
        "visual-feedback-output-json-schema-v1"
    ] = QWEN38_FEEDBACK_JSON_SCHEMA_POLICY_VERSION
    requested_json_schema_name: Literal["visual_feedback_output_v1"] = (
        QWEN38_FEEDBACK_JSON_SCHEMA_NAME
    )
    requested_json_schema_sha256: Literal[QWEN38_FEEDBACK_JSON_SCHEMA_SHA256] = (
        QWEN38_FEEDBACK_JSON_SCHEMA_SHA256
    )
    requested_json_schema_strict: Literal[True] = True
    max_tokens: None = None
    max_completion_tokens: Literal[6144] = QWEN38_FEEDBACK_MAX_COMPLETION_TOKENS_V2
    output_token_reservation_ceiling_per_call: Literal[6154] = (
        QWEN38_FEEDBACK_OUTPUT_RESERVATION_TOKENS_V2
    )
    timeout_seconds: Literal[600] = QWEN38_FEEDBACK_TIMEOUT_SECONDS
    requested_stream: Literal[False] = False
    temperature: None = None
    top_p: None = None
    seed: None = None
    provider_internal_max_attempts: Literal[1] = 1
    concurrency: Literal[2] = 2
    wire_kind: Literal["round3_primary_json_schema_v1"] = (
        QWEN38_FEEDBACK_ROUND3_WIRE_KIND_V2
    )
    transport_policy_version: Literal[
        "visual-feedback-qwen38-dashscope-json-schema-round3-primary-v1"
    ] = ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_VERSION_V1
    transport_policy_sha256: Literal[
        ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    ] = ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    outer_orchestration_policy_version: Literal[
        "portfolio-s1-feedback-round3-global-retry-v2"
    ] = QWEN38_FEEDBACK_ROUND3_RETRY_POLICY_VERSION_V2
    outer_orchestration_policy_sha256: Literal[
        QWEN38_FEEDBACK_ROUND3_RETRY_POLICY_SHA256_V2
    ] = QWEN38_FEEDBACK_ROUND3_RETRY_POLICY_SHA256_V2
    prompt_policy_version: Literal["visual-feedback-gcs-policy-labels-prompt-v6"] = (
        QWEN38_FEEDBACK_ROUND3_PROMPT_POLICY_VERSION_V6
    )
    prompt_policy_sha256: Literal[QWEN38_FEEDBACK_ROUND3_PROMPT_POLICY_SHA256_V6] = (
        QWEN38_FEEDBACK_ROUND3_PROMPT_POLICY_SHA256_V6
    )
    parser_policy_version: Literal["visual-feedback-free-text-trim-v3"] = (
        QWEN38_FEEDBACK_ROUND3_PARSER_POLICY_VERSION_V3
    )
    parser_policy_sha256: Literal[QWEN38_FEEDBACK_ROUND3_PARSER_POLICY_SHA256_V3] = (
        QWEN38_FEEDBACK_ROUND3_PARSER_POLICY_SHA256_V3
    )
    cache_namespace: Literal["feedback-evaluator-v14"] = (
        QWEN38_FEEDBACK_CACHE_NAMESPACE_V4
    )
    result_schema_version: Literal[8] = QWEN38_FEEDBACK_RESULT_SCHEMA_VERSION_V4
    bound_artifact_schema_version: Literal[2] = (
        QWEN38_FEEDBACK_BOUND_ARTIFACT_SCHEMA_VERSION_V4
    )
    bound_artifact_policy_version: Literal[
        "portfolio-s1-bound-feedback-round3-schema-v1"
    ] = QWEN38_FEEDBACK_BOUND_ARTIFACT_POLICY_VERSION_V4
    fixed_selected_query_count: Literal[240] = QWEN38_FEEDBACK_SELECTED_COUNT
    live_canary_selected_query_count: Literal[12] = (
        QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_SELECTED_COUNT
    )
    provider_call_ceiling: Literal[15] = (
        QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_PROVIDER_CALL_CEILING
    )
    future_full_provider_call_ceiling: Literal[243] = (
        QWEN38_FEEDBACK_PROVIDER_CALL_CEILING_V3
    )
    live_call_authority: Literal[True] = True
    live_call_authority_scope: Literal["canary12_plus_three_global_retries_only"] = (
        "canary12_plus_three_global_retries_only"
    )
    full_run_live_authorized: Literal[False] = False
    future_full_owner_budget_authorization_status: Literal["not_granted"] = (
        "not_granted"
    )
    phase60_requires_new_owner_approval: Literal[True] = True
    image_source_membership_ancestry_only: Literal[True] = True
    old_remote_auth_does_not_authorize_round3_transport: Literal[True] = True
    historical_feedback_outputs_imported: Literal[0] = 0
    terminated_json_object_canary_outputs_imported: Literal[0] = 0
    round3_live_authority_requires_exact_v4_v7_v14_triad: Literal[True] = True

    @model_validator(mode="after")
    def _validate_v4_lock(self) -> Self:
        if (
            self.requested_response_format != "json_schema"
            or self.structured_output_response_format != "json_schema"
            or not self.structured_output_json_schema_used
            or not self.requested_json_schema
            or not self.requested_json_schema_strict
            or self.requested_json_schema_sha256 != QWEN38_FEEDBACK_JSON_SCHEMA_SHA256
            or self.outer_orchestration_policy_sha256
            != QWEN38_FEEDBACK_ROUND3_RETRY_POLICY_SHA256_V2
            or self.concurrency != 2
            or self.provider_call_ceiling != 15
            or self.future_full_provider_call_ceiling != 243
            or self.full_run_live_authorized
            or self.future_full_owner_budget_authorization_status != "not_granted"
            or not self.phase60_requires_new_owner_approval
            or self.historical_feedback_outputs_imported != 0
            or self.terminated_json_object_canary_outputs_imported != 0
            or not self.old_remote_auth_does_not_authorize_round3_transport
        ):
            raise ValueError("Qwen3.8 Feedback Round3 schema source lock drifted")
        if self.source_lock_sha256 != _self_hash(self, "source_lock_sha256"):
            raise ValueError("Qwen3.8 Feedback model source lock v4 self hash mismatch")
        return self


class Qwen38FeedbackModelSourceLockV5(Qwen38FeedbackModelSourceLockV4):
    """Owner-approved live phase60 identity over the completed canary prefix."""

    schema_version: Literal[5] = 5
    policy_version: Literal["portfolio-s1-qwen38-feedback-model-source-lock-v5"] = (
        QWEN38_FEEDBACK_SOURCE_LOCK_POLICY_VERSION_V5
    )
    outer_orchestration_policy_version: Literal[
        "portfolio-s1-feedback-round3-phase60-global-retry-v1"
    ] = ROUND3_PHASE60_RETRY_POLICY_VERSION_V1
    outer_orchestration_policy_sha256: Literal[
        ROUND3_PHASE60_RETRY_POLICY_SHA256_V1
    ] = ROUND3_PHASE60_RETRY_POLICY_SHA256_V1
    provider_call_ceiling: Literal[60] = (
        QWEN38_FEEDBACK_PHASE60_NEW_PROVIDER_CALL_CEILING
    )
    cumulative_provider_call_ceiling: Literal[75] = (
        QWEN38_FEEDBACK_PHASE60_CUMULATIVE_PROVIDER_CALL_CEILING
    )
    phase60_selected_query_count: Literal[60] = 60
    phase60_prefix_selected_count: Literal[12] = (
        QWEN38_FEEDBACK_PHASE60_PREFIX_SELECTED_COUNT
    )
    phase60_new_first_call_count: Literal[48] = (
        QWEN38_FEEDBACK_PHASE60_NEW_FIRST_CALL_COUNT
    )
    phase60_new_retry_token_count: Literal[12] = (
        QWEN38_FEEDBACK_PHASE60_NEW_RETRY_TOKEN_COUNT
    )
    canary_prefix_provider_call_count: Literal[15] = (
        QWEN38_FEEDBACK_PHASE60_PREFIX_PROVIDER_CALL_COUNT
    )
    canary_prefix_retry_count: Literal[3] = QWEN38_FEEDBACK_PHASE60_PREFIX_RETRY_COUNT
    canary_prefix_retry_tokens_reusable: Literal[False] = False
    canary_prefix_run_file_sha256: Literal[
        QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_RUN_FILE_SHA256
    ] = QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_RUN_FILE_SHA256
    canary_prefix_run_sha256: Literal[
        QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_RUN_SHA256
    ] = QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_RUN_SHA256
    canary_prefix_artifact_set_sha256: Literal[
        QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_ARTIFACT_SET_SHA256
    ] = QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_ARTIFACT_SET_SHA256
    canary_prefix_selection_sha256: Literal[
        QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_SELECTION_SHA256
    ] = QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_SELECTION_SHA256
    live_call_authority: Literal[True] = True
    live_provider_calls_authorized: Literal[True] = True
    live_call_authority_scope: Literal[
        "phase60_prefix12_plus_48_first_plus_up_to_12_retries_only"
    ] = "phase60_prefix12_plus_48_first_plus_up_to_12_retries_only"
    owner_phase60_budget_authorization_status: Literal["granted"] = "granted"
    owner_phase60_retry_authorization_status: Literal["granted"] = "granted"
    owner_approval_decision_source: Literal["current_user_instruction"] = (
        "current_user_instruction"
    )
    owner_approved_on: Literal["2026-08-11"] = "2026-08-11"
    fresh_maximum_reservation_cny: Literal["27.692640000000"] = (
        QWEN38_FEEDBACK_PHASE60_FRESH_MAXIMUM_RESERVATION_CNY
    )
    fresh_technical_hard_cap_cny: Literal["28.000000000000"] = (
        QWEN38_FEEDBACK_PHASE60_FRESH_TECHNICAL_HARD_CAP_CNY
    )
    cumulative_technical_hard_cap_cny: Literal["54.264100000000"] = (
        QWEN38_FEEDBACK_PHASE60_CUMULATIVE_TECHNICAL_HARD_CAP_CNY
    )
    phase60_requires_new_owner_approval: Literal[False] = False
    phase120_requires_new_owner_approval: Literal[True] = True
    full_run_live_authorized: Literal[False] = False
    future_full_owner_budget_authorization_status: Literal["not_granted"] = (
        "not_granted"
    )
    historical_feedback_outputs_imported: Literal[12] = 12
    terminated_json_object_canary_outputs_imported: Literal[0] = 0
    round3_live_authority_requires_exact_v4_v7_v14_triad: Literal[False] = False
    phase60_live_identity_requires_exact_v5_v8_v15_triad: Literal[True] = True

    @model_validator(mode="after")
    def _validate_v4_lock(self) -> Self:
        if (
            self.requested_response_format != "json_schema"
            or self.structured_output_response_format != "json_schema"
            or not self.structured_output_json_schema_used
            or not self.requested_json_schema
            or not self.requested_json_schema_strict
            or self.requested_json_schema_sha256 != QWEN38_FEEDBACK_JSON_SCHEMA_SHA256
            or self.outer_orchestration_policy_sha256
            != ROUND3_PHASE60_RETRY_POLICY_SHA256_V1
            or self.concurrency != 2
            or self.phase60_prefix_selected_count + self.phase60_new_first_call_count
            != self.phase60_selected_query_count
            or self.phase60_new_first_call_count + self.phase60_new_retry_token_count
            != self.provider_call_ceiling
            or self.canary_prefix_provider_call_count + self.provider_call_ceiling
            != self.cumulative_provider_call_ceiling
            or self.canary_prefix_retry_count != 3
            or self.canary_prefix_retry_tokens_reusable
            or not self.live_call_authority
            or not self.live_provider_calls_authorized
            or self.owner_phase60_budget_authorization_status != "granted"
            or self.owner_phase60_retry_authorization_status != "granted"
            or self.owner_approval_decision_source != "current_user_instruction"
            or self.owner_approved_on != "2026-08-11"
            or self.fresh_maximum_reservation_cny
            != QWEN38_FEEDBACK_PHASE60_FRESH_MAXIMUM_RESERVATION_CNY
            or self.fresh_technical_hard_cap_cny
            != QWEN38_FEEDBACK_PHASE60_FRESH_TECHNICAL_HARD_CAP_CNY
            or self.cumulative_technical_hard_cap_cny
            != QWEN38_FEEDBACK_PHASE60_CUMULATIVE_TECHNICAL_HARD_CAP_CNY
            or self.phase60_requires_new_owner_approval
            or not self.phase120_requires_new_owner_approval
            or self.full_run_live_authorized
            or self.historical_feedback_outputs_imported != 12
            or self.terminated_json_object_canary_outputs_imported != 0
            or not self.old_remote_auth_does_not_authorize_round3_transport
            or self.round3_live_authority_requires_exact_v4_v7_v14_triad
        ):
            raise ValueError("Qwen3.8 Feedback phase60 live source lock drifted")
        if self.source_lock_sha256 != _self_hash(self, "source_lock_sha256"):
            raise ValueError("Qwen3.8 Feedback model source lock v5 self hash mismatch")
        return self


class Qwen37FeedbackPricingLockV1(_StrictFrozenModel):
    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-qwen37-feedback-pricing-lock"] = (
        "portfolio-s1-qwen37-feedback-pricing-lock"
    )
    policy_version: Literal["portfolio-s1-qwen37-feedback-pricing-lock-v1"] = (
        QWEN37_FEEDBACK_PRICING_LOCK_POLICY_VERSION
    )
    provider: Literal["qwen"] = "qwen"
    model: Literal["qwen3.7-plus-2026-05-26"] = QWEN37_FEEDBACK_MODEL
    deployment_region: Literal["china-beijing"] = "china-beijing"
    pricing_mode: Literal["thinking-realtime-list-price"] = (
        "thinking-realtime-list-price"
    )
    currency: Literal["CNY"] = "CNY"
    tier_min_input_tokens_exclusive: Literal[0] = 0
    tier_max_input_tokens_inclusive: Literal[256000] = (
        QWEN37_FEEDBACK_PRICING_TIER_MAX_INPUT_TOKENS
    )
    input_cny_per_million_tokens: Literal[2] = (
        QWEN37_FEEDBACK_INPUT_CNY_PER_MILLION_TOKENS
    )
    output_cny_per_million_tokens: Literal[8] = (
        QWEN37_FEEDBACK_OUTPUT_CNY_PER_MILLION_TOKENS
    )
    reasoning_tokens_billed_as_output_tokens: Literal[True] = True
    thinking_and_answer_output_rate_equal: Literal[True] = True
    wire_max_completion_tokens: Literal[4096] = QWEN37_FEEDBACK_MAX_COMPLETION_TOKENS
    max_completion_tokens_documented_upper_tolerance_tokens: Literal[10] = (
        QWEN37_FEEDBACK_MAX_COMPLETION_TOKENS_TOLERANCE
    )
    input_token_reservation_ceiling_per_call: Literal[20000] = (
        QWEN37_FEEDBACK_INPUT_RESERVATION_TOKENS
    )
    output_token_reservation_ceiling_per_call: Literal[4106] = (
        QWEN37_FEEDBACK_OUTPUT_RESERVATION_TOKENS
    )
    per_call_reservation_cny: Literal["0.072848000000"] = (
        QWEN37_FEEDBACK_PER_CALL_RESERVATION_CNY
    )
    provider_call_ceiling: Literal[48] = QWEN37_FEEDBACK_PROVIDER_CALL_CEILING
    maximum_reservation_cny: Literal["3.496704000000"] = (
        QWEN37_FEEDBACK_MAXIMUM_RESERVATION_CNY
    )
    phase_hard_cap_cny: Literal["4.000000000000"] = QWEN37_FEEDBACK_PHASE_HARD_CAP_CNY
    discounts_assumed: Literal[False] = False
    free_quota_assumed: Literal[False] = False
    context_cache_assumed: Literal[False] = False
    batch_discount_assumed: Literal[False] = False
    source_url: Literal["https://help.aliyun.com/zh/model-studio/model-pricing"] = (
        "https://help.aliyun.com/zh/model-studio/model-pricing"
    )
    source_retrieved_on: Literal["2026-08-10"] = "2026-08-10"
    pricing_lock_sha256: Sha256

    @model_validator(mode="after")
    def _validate_lock(self) -> Self:
        if self.pricing_lock_sha256 != _self_hash(self, "pricing_lock_sha256"):
            raise ValueError("Qwen Feedback pricing lock self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class Qwen37FeedbackPricingLockV2(_StrictFrozenModel):
    """Forward-only exact240 reservation envelope over unchanged list pricing."""

    schema_version: Literal[2] = 2
    kind: Literal["portfolio-s1-qwen37-feedback-pricing-lock"] = (
        "portfolio-s1-qwen37-feedback-pricing-lock"
    )
    policy_version: Literal["portfolio-s1-qwen37-feedback-pricing-lock-v2"] = (
        QWEN37_FEEDBACK_PRICING_LOCK_POLICY_VERSION_V2
    )
    provider: Literal["qwen"] = "qwen"
    model: Literal["qwen3.7-plus-2026-05-26"] = QWEN37_FEEDBACK_MODEL
    deployment_region: Literal["china-beijing"] = "china-beijing"
    pricing_mode: Literal["thinking-realtime-list-price"] = (
        "thinking-realtime-list-price"
    )
    currency: Literal["CNY"] = "CNY"
    tier_min_input_tokens_exclusive: Literal[0] = 0
    tier_max_input_tokens_inclusive: Literal[256000] = (
        QWEN37_FEEDBACK_PRICING_TIER_MAX_INPUT_TOKENS
    )
    input_cny_per_million_tokens: Literal[2] = (
        QWEN37_FEEDBACK_INPUT_CNY_PER_MILLION_TOKENS
    )
    output_cny_per_million_tokens: Literal[8] = (
        QWEN37_FEEDBACK_OUTPUT_CNY_PER_MILLION_TOKENS
    )
    reasoning_tokens_billed_as_output_tokens: Literal[True] = True
    thinking_and_answer_output_rate_equal: Literal[True] = True
    wire_max_completion_tokens: Literal[4096] = QWEN37_FEEDBACK_MAX_COMPLETION_TOKENS
    max_completion_tokens_documented_upper_tolerance_tokens: Literal[10] = (
        QWEN37_FEEDBACK_MAX_COMPLETION_TOKENS_TOLERANCE
    )
    input_token_reservation_ceiling_per_call: Literal[20000] = (
        QWEN37_FEEDBACK_INPUT_RESERVATION_TOKENS
    )
    output_token_reservation_ceiling_per_call: Literal[4106] = (
        QWEN37_FEEDBACK_OUTPUT_RESERVATION_TOKENS
    )
    per_call_reservation_cny: Literal["0.072848000000"] = (
        QWEN37_FEEDBACK_PER_CALL_RESERVATION_CNY
    )
    provider_call_ceiling: Literal[240] = QWEN37_FEEDBACK_PROVIDER_CALL_CEILING_V2
    maximum_reservation_cny: Literal["17.483520000000"] = (
        QWEN37_FEEDBACK_MAXIMUM_RESERVATION_CNY_V2
    )
    phase_hard_cap_cny: Literal["18.000000000000"] = (
        QWEN37_FEEDBACK_PHASE_HARD_CAP_CNY_V2
    )
    discounts_assumed: Literal[False] = False
    free_quota_assumed: Literal[False] = False
    context_cache_assumed: Literal[False] = False
    batch_discount_assumed: Literal[False] = False
    source_url: Literal["https://help.aliyun.com/zh/model-studio/model-pricing"] = (
        "https://help.aliyun.com/zh/model-studio/model-pricing"
    )
    source_retrieved_on: Literal["2026-08-10"] = "2026-08-10"
    pricing_lock_sha256: Sha256

    @model_validator(mode="after")
    def _validate_lock(self) -> Self:
        if self.pricing_lock_sha256 != _self_hash(self, "pricing_lock_sha256"):
            raise ValueError("Qwen Feedback pricing lock v2 self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class Qwen38FeedbackPricingLockV3(_StrictFrozenModel):
    """Owner-approved technical reservation envelope for exact Discovery240."""

    schema_version: Literal[3] = 3
    kind: Literal["portfolio-s1-qwen38-feedback-pricing-lock"] = (
        "portfolio-s1-qwen38-feedback-pricing-lock"
    )
    policy_version: Literal["portfolio-s1-qwen38-feedback-pricing-lock-v3"] = (
        QWEN38_FEEDBACK_PRICING_LOCK_POLICY_VERSION_V3
    )
    provider: Literal["qwen"] = QWEN38_FEEDBACK_PROVIDER
    model: Literal["qwen3.8-max"] = QWEN38_FEEDBACK_MODEL
    deployment_region: Literal["china-beijing"] = "china-beijing"
    pricing_mode: Literal["thinking-realtime-list-price"] = (
        "thinking-realtime-list-price"
    )
    currency: Literal["CNY"] = "CNY"
    tier_min_input_tokens_exclusive: Literal[0] = 0
    tier_max_input_tokens_inclusive: Literal[1000000] = (
        QWEN38_FEEDBACK_PRICING_TIER_MAX_INPUT_TOKENS
    )
    input_cny_per_million_tokens: Literal[12] = (
        QWEN38_FEEDBACK_INPUT_CNY_PER_MILLION_TOKENS
    )
    output_cny_per_million_tokens: Literal[36] = (
        QWEN38_FEEDBACK_OUTPUT_CNY_PER_MILLION_TOKENS
    )
    reasoning_tokens_billed_as_output_tokens: Literal[True] = True
    thinking_and_answer_output_rate_equal: Literal[True] = True
    wire_max_completion_tokens: Literal[4096] = QWEN38_FEEDBACK_MAX_COMPLETION_TOKENS
    max_completion_tokens_documented_upper_tolerance_tokens: Literal[10] = (
        QWEN38_FEEDBACK_MAX_COMPLETION_TOKENS_TOLERANCE
    )
    input_token_reservation_ceiling_per_call: Literal[20000] = (
        QWEN38_FEEDBACK_INPUT_RESERVATION_TOKENS
    )
    output_token_reservation_ceiling_per_call: Literal[4106] = (
        QWEN38_FEEDBACK_OUTPUT_RESERVATION_TOKENS
    )
    per_call_reservation_cny: Literal["0.387816000000"] = (
        QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY
    )
    provider_call_ceiling: Literal[240] = QWEN38_FEEDBACK_PROVIDER_CALL_CEILING
    maximum_reservation_cny: Literal["93.075840000000"] = (
        QWEN38_FEEDBACK_MAXIMUM_RESERVATION_CNY
    )
    technical_phase_hard_cap_cny: Literal["94.000000000000"] = (
        QWEN38_FEEDBACK_TECHNICAL_PHASE_HARD_CAP_CNY
    )
    live_use_requires_separate_owner_budget_authorization: Literal[True] = True
    owner_budget_authorization_decision_source: Literal["current_user_instruction"] = (
        "current_user_instruction"
    )
    owner_budget_authorization_scope: Literal[
        "core-opt800-s1-feedback-discovery-selected-240-assets"
    ] = "core-opt800-s1-feedback-discovery-selected-240-assets"
    owner_budget_authorization_status: Literal[
        "granted_for_exact_discovery_selected240"
    ] = "granted_for_exact_discovery_selected240"
    owner_budget_authorized_cap_cny: Literal["94.000000000000"] = (
        QWEN38_FEEDBACK_TECHNICAL_PHASE_HARD_CAP_CNY
    )
    owner_budget_authorized_on: Literal["2026-08-10"] = "2026-08-10"
    discounts_assumed: Literal[False] = False
    free_quota_assumed: Literal[False] = False
    context_cache_assumed: Literal[False] = False
    batch_discount_assumed: Literal[False] = False
    source_url: Literal["https://help.aliyun.com/zh/model-studio/model-pricing"] = (
        "https://help.aliyun.com/zh/model-studio/model-pricing"
    )
    source_retrieved_on: Literal["2026-08-10"] = "2026-08-10"
    pricing_lock_sha256: Sha256

    @model_validator(mode="after")
    def _validate_lock(self) -> Self:
        if self.pricing_lock_sha256 != _self_hash(self, "pricing_lock_sha256"):
            raise ValueError("Qwen3.8 Feedback pricing lock v3 self hash mismatch")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class Qwen38FeedbackPricingLockV4(Qwen38FeedbackPricingLockV3):
    """Exact240 budget plus one owner-approved global schema retry token."""

    schema_version: Literal[4] = 4
    policy_version: Literal["portfolio-s1-qwen38-feedback-pricing-lock-v4"] = (
        QWEN38_FEEDBACK_PRICING_LOCK_POLICY_VERSION_V4
    )
    provider_call_ceiling: Literal[241] = QWEN38_FEEDBACK_PROVIDER_CALL_CEILING_V2
    selected_query_count: Literal[240] = QWEN38_FEEDBACK_SELECTED_COUNT
    normal_attempts_per_selected_query: Literal[1] = 1
    global_retry_token_count: Literal[1] = 1
    max_attempts_per_retried_query: Literal[2] = 2
    retry_policy: Literal["one_global_same_entry_strict_schema_retry_v1"] = (
        "one_global_same_entry_strict_schema_retry_v1"
    )
    retry_eligible_error_codes: tuple[Literal["invalid_feedback_json"], ...] = (
        "invalid_feedback_json",
    )
    attempt_transport_policy_version: Literal[
        "visual-feedback-qwen38-dashscope-json-schema-v6"
    ] = QWEN38_FEEDBACK_TRANSPORT_POLICY_VERSION
    attempt_transport_retry_policy: Literal[
        "no_internal_retry_each_provider_attempt"
    ] = "no_internal_retry_each_provider_attempt"
    maximum_reservation_cny: Literal["93.463656000000"] = (
        QWEN38_FEEDBACK_MAXIMUM_RESERVATION_CNY_V2
    )
    owner_budget_authorization_scope: Literal[
        "core-opt800-s1-feedback-discovery-selected-240-assets-plus-one-global-"
        "same-entry-invalid-json-retry"
    ] = (
        "core-opt800-s1-feedback-discovery-selected-240-assets-plus-one-global-"
        "same-entry-invalid-json-retry"
    )
    owner_budget_authorization_status: Literal[
        "granted_for_exact_discovery_selected240_plus_one_global_invalid_json_retry"
    ] = "granted_for_exact_discovery_selected240_plus_one_global_invalid_json_retry"

    @field_validator("retry_eligible_error_codes", mode="before")
    @classmethod
    def _retry_error_codes_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_v4_lock(self) -> Self:
        if self.retry_eligible_error_codes != ("invalid_feedback_json",):
            raise ValueError("Qwen3.8 Feedback pricing v4 retry scope drifted")
        if self.pricing_lock_sha256 != _self_hash(self, "pricing_lock_sha256"):
            raise ValueError("Qwen3.8 Feedback pricing lock v4 self hash mismatch")
        return self


class Qwen38FeedbackPricingLockV5(Qwen38FeedbackPricingLockV3):
    """Live fresh-v3 envelope under CNY150 authority and a CNY113 stop."""

    schema_version: Literal[5] = 5
    policy_version: Literal["portfolio-s1-qwen38-feedback-pricing-lock-v5"] = (
        QWEN38_FEEDBACK_PRICING_LOCK_POLICY_VERSION_V5
    )
    wire_max_completion_tokens: Literal[6144] = QWEN38_FEEDBACK_MAX_COMPLETION_TOKENS_V2
    output_token_reservation_ceiling_per_call: Literal[6154] = (
        QWEN38_FEEDBACK_OUTPUT_RESERVATION_TOKENS_V2
    )
    per_call_reservation_cny: Literal["0.461544000000"] = (
        QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY_V2
    )
    provider_call_ceiling: Literal[243] = QWEN38_FEEDBACK_PROVIDER_CALL_CEILING_V3
    selected_query_count: Literal[240] = QWEN38_FEEDBACK_SELECTED_COUNT
    normal_attempts_per_selected_query: Literal[1] = 1
    global_retry_token_count: Literal[3] = 3
    max_attempts_per_retried_query: Literal[2] = 2
    retry_policy: Literal["three_global_same_entry_schema_or_length_retries_v2"] = (
        "three_global_same_entry_schema_or_length_retries_v2"
    )
    retry_eligible_error_codes: tuple[Literal["invalid_feedback_json"], ...] = (
        "invalid_feedback_json",
    )
    retry_eligible_finish_reasons: tuple[Literal["stop", "length"], ...] = (
        "stop",
        "length",
    )
    outer_orchestration_policy_version: Literal[
        "portfolio-s1-feedback-global-schema-or-length-retry-v2"
    ] = QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_VERSION_V2
    outer_orchestration_policy_sha256: Literal[
        QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V2
    ] = QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V2
    attempt_transport_policy_version: Literal[
        "visual-feedback-qwen38-dashscope-json-schema-v7"
    ] = QWEN38_FEEDBACK_TRANSPORT_POLICY_VERSION_V7
    attempt_transport_policy_sha256: Literal[
        QWEN38_FEEDBACK_TRANSPORT_POLICY_SHA256_V7
    ] = QWEN38_FEEDBACK_TRANSPORT_POLICY_SHA256_V7
    attempt_transport_retry_policy: Literal[
        "no_internal_retry_each_provider_attempt"
    ] = "no_internal_retry_each_provider_attempt"
    maximum_reservation_cny: Literal["112.155192000000"] = (
        QWEN38_FEEDBACK_MAXIMUM_RESERVATION_CNY_V3
    )
    technical_phase_hard_cap_cny: Literal["113.000000000000"] = (
        QWEN38_FEEDBACK_TECHNICAL_PHASE_HARD_CAP_CNY_V2
    )
    fresh_run_and_retry_scope_owner_approved: Literal[True] = True
    fresh_run_and_retry_scope_owner_approved_on: Literal["2026-08-10"] = "2026-08-10"
    live_provider_calls_authorized: Literal[True] = True
    live_use_requires_separate_owner_budget_authorization: Literal[False] = False
    owner_budget_authorization_scope: Literal[
        "core-opt800-s1-feedback-discovery-selected-240-assets-plus-three-global-"
        "same-entry-schema-or-length-retries"
    ] = (
        "core-opt800-s1-feedback-discovery-selected-240-assets-plus-three-global-"
        "same-entry-schema-or-length-retries"
    )
    owner_budget_authorization_status: Literal[
        "granted_for_exact_discovery_selected240_plus_three_global_schema_or_"
        "length_retries_under_cny150_owner_ceiling"
    ] = (
        "granted_for_exact_discovery_selected240_plus_three_global_schema_or_"
        "length_retries_under_cny150_owner_ceiling"
    )
    owner_budget_authorized_cap_cny: Literal["150.000000000000"] = (
        QWEN38_FEEDBACK_OWNER_AUTHORIZED_BUDGET_CEILING_CNY
    )
    owner_budget_authorized_on: Literal["2026-08-10"] = "2026-08-10"

    @field_validator(
        "retry_eligible_error_codes",
        "retry_eligible_finish_reasons",
        mode="before",
    )
    @classmethod
    def _retry_eligibility_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_v5_lock(self) -> Self:
        if (
            self.retry_eligible_error_codes != ("invalid_feedback_json",)
            or self.retry_eligible_finish_reasons != ("stop", "length")
            or Decimal(self.per_call_reservation_cny) * self.provider_call_ceiling
            != Decimal(self.maximum_reservation_cny)
            or Decimal(self.maximum_reservation_cny)
            >= Decimal(self.technical_phase_hard_cap_cny)
            or Decimal(self.technical_phase_hard_cap_cny)
            > Decimal(self.owner_budget_authorized_cap_cny)
        ):
            raise ValueError("Qwen3.8 Feedback pricing v5 envelope drifted")
        if self.pricing_lock_sha256 != _self_hash(self, "pricing_lock_sha256"):
            raise ValueError("Qwen3.8 Feedback pricing lock v5 self hash mismatch")
        return self


class Qwen38FeedbackPricingLockV6(Qwen38FeedbackPricingLockV3):
    """Cumulative Round3 envelope for a wholly fresh fixed-240 run."""

    schema_version: Literal[6] = 6
    policy_version: Literal["portfolio-s1-qwen38-feedback-pricing-lock-v6"] = (
        QWEN38_FEEDBACK_PRICING_LOCK_POLICY_VERSION_V6
    )
    wire_max_completion_tokens: Literal[6144] = QWEN38_FEEDBACK_MAX_COMPLETION_TOKENS_V2
    output_token_reservation_ceiling_per_call: Literal[6154] = (
        QWEN38_FEEDBACK_OUTPUT_RESERVATION_TOKENS_V2
    )
    per_call_reservation_cny: Literal["0.461544000000"] = (
        QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY_V2
    )
    provider_call_ceiling: Literal[243] = QWEN38_FEEDBACK_PROVIDER_CALL_CEILING_V3
    selected_query_count: Literal[240] = QWEN38_FEEDBACK_SELECTED_COUNT
    phase_counts: tuple[Literal[12, 60, 120, 240], ...] = (12, 60, 120, 240)
    normal_attempts_per_selected_query: Literal[1] = 1
    global_retry_token_count: Literal[3] = 3
    provider_internal_max_attempts: Literal[1] = 1
    max_lifetime_attempts_per_retried_query: Literal[2] = 2
    retry_policy: Literal[
        "three_global_same_entry_json_object_parser_or_length_retries_round3_v1"
    ] = "three_global_same_entry_json_object_parser_or_length_retries_round3_v1"
    retry_eligible_error_codes: tuple[Literal["invalid_feedback_json"], ...] = (
        "invalid_feedback_json",
    )
    retry_eligible_finish_reasons: tuple[Literal["stop", "length"], ...] = (
        "stop",
        "length",
    )
    outer_orchestration_policy_version: Literal[
        "portfolio-s1-feedback-round3-global-retry-v1"
    ] = QWEN38_FEEDBACK_ROUND3_RETRY_POLICY_VERSION_V1
    outer_orchestration_policy_sha256: Literal[
        QWEN38_FEEDBACK_ROUND3_RETRY_POLICY_SHA256_V1
    ] = QWEN38_FEEDBACK_ROUND3_RETRY_POLICY_SHA256_V1
    attempt_transport_policy_version: Literal[
        "visual-feedback-qwen38-dashscope-json-object-round3-primary-v1"
    ] = ROUND3_PRIMARY_TRANSPORT_POLICY_VERSION_V1
    attempt_transport_policy_sha256: Literal[
        ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1
    ] = ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1
    attempt_response_format: Literal["json_object"] = "json_object"
    attempt_transport_retry_policy: Literal[
        "no_internal_retry_each_provider_attempt"
    ] = "no_internal_retry_each_provider_attempt"
    maximum_reservation_cny: Literal["112.155192000000"] = (
        QWEN38_FEEDBACK_MAXIMUM_RESERVATION_CNY_V3
    )
    prior_cumulative_actual_cost_cny: Literal["22.764432000000"] = (
        QWEN38_FEEDBACK_ROUND3_PRIOR_ACTUAL_COST_CNY
    )
    cumulative_maximum_reservation_cny: Literal["134.919624000000"] = (
        QWEN38_FEEDBACK_ROUND3_CUMULATIVE_MAXIMUM_CNY
    )
    technical_phase_hard_cap_cny: Literal["135.000000000000"] = (
        QWEN38_FEEDBACK_ROUND3_CUMULATIVE_TECHNICAL_HARD_CAP_CNY
    )
    technical_cap_scope: Literal["prior_actual_plus_fresh_reservations"] = (
        "prior_actual_plus_fresh_reservations"
    )
    fresh_run_and_retry_scope_owner_approved: Literal[True] = True
    live_provider_calls_authorized: Literal[True] = True
    live_use_requires_separate_owner_budget_authorization: Literal[False] = False
    owner_budget_authorization_scope: Literal[
        "round3-wholly-fresh-fixed-discovery240-json-object-plus-three-global-"
        "same-entry-parser-or-length-retries"
    ] = (
        "round3-wholly-fresh-fixed-discovery240-json-object-plus-three-global-"
        "same-entry-parser-or-length-retries"
    )
    owner_budget_authorization_status: Literal[
        "granted_round3_json_object_under_cny150_owner_ceiling_and_cny135_"
        "cumulative_technical_stop"
    ] = (
        "granted_round3_json_object_under_cny150_owner_ceiling_and_cny135_"
        "cumulative_technical_stop"
    )
    owner_budget_authorized_cap_cny: Literal["150.000000000000"] = (
        QWEN38_FEEDBACK_OWNER_AUTHORIZED_BUDGET_CEILING_CNY
    )
    owner_budget_authorized_on: Literal["2026-08-11"] = "2026-08-11"
    historical_feedback_outputs_imported: Literal[0] = 0

    @field_validator(
        "phase_counts",
        "retry_eligible_error_codes",
        "retry_eligible_finish_reasons",
        mode="before",
    )
    @classmethod
    def _round3_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_v6_lock(self) -> Self:
        if (
            self.phase_counts != (12, 60, 120, 240)
            or self.retry_eligible_error_codes != ("invalid_feedback_json",)
            or self.retry_eligible_finish_reasons != ("stop", "length")
            or Decimal(self.per_call_reservation_cny) * self.provider_call_ceiling
            != Decimal(self.maximum_reservation_cny)
            or Decimal(self.prior_cumulative_actual_cost_cny)
            + Decimal(self.maximum_reservation_cny)
            != Decimal(self.cumulative_maximum_reservation_cny)
            or Decimal(self.cumulative_maximum_reservation_cny)
            >= Decimal(self.technical_phase_hard_cap_cny)
            or Decimal(self.technical_phase_hard_cap_cny)
            > Decimal(self.owner_budget_authorized_cap_cny)
            or not self.live_provider_calls_authorized
        ):
            raise ValueError("Qwen3.8 Feedback pricing v6 envelope drifted")
        if self.pricing_lock_sha256 != _self_hash(self, "pricing_lock_sha256"):
            raise ValueError("Qwen3.8 Feedback pricing lock v6 self hash mismatch")
        return self


class Qwen38FeedbackPricingLockV7(Qwen38FeedbackPricingLockV3):
    """Live schema-canary budget plus a non-live future full-run envelope."""

    schema_version: Literal[7] = 7
    policy_version: Literal["portfolio-s1-qwen38-feedback-pricing-lock-v7"] = (
        QWEN38_FEEDBACK_PRICING_LOCK_POLICY_VERSION_V7
    )
    wire_max_completion_tokens: Literal[6144] = QWEN38_FEEDBACK_MAX_COMPLETION_TOKENS_V2
    output_token_reservation_ceiling_per_call: Literal[6154] = (
        QWEN38_FEEDBACK_OUTPUT_RESERVATION_TOKENS_V2
    )
    per_call_reservation_cny: Literal["0.461544000000"] = (
        QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY_V2
    )
    provider_call_ceiling: Literal[15] = (
        QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_PROVIDER_CALL_CEILING
    )
    future_full_provider_call_ceiling: Literal[243] = (
        QWEN38_FEEDBACK_PROVIDER_CALL_CEILING_V3
    )
    selected_query_count: Literal[240] = QWEN38_FEEDBACK_SELECTED_COUNT
    live_authorized_selected_query_count: Literal[12] = (
        QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_SELECTED_COUNT
    )
    phase_counts: tuple[Literal[12, 60, 120, 240], ...] = (12, 60, 120, 240)
    live_authorized_phase_counts: tuple[Literal[12], ...] = (12,)
    concurrency: Literal[2] = 2
    normal_attempts_per_selected_query: Literal[1] = 1
    global_retry_token_count: Literal[3] = 3
    provider_internal_max_attempts: Literal[1] = 1
    max_lifetime_attempts_per_retried_query: Literal[2] = 2
    retry_policy: Literal[
        "three_global_same_entry_json_schema_parser_or_length_retries_round3_v2"
    ] = "three_global_same_entry_json_schema_parser_or_length_retries_round3_v2"
    retry_eligible_error_codes: tuple[Literal["invalid_feedback_json"], ...] = (
        "invalid_feedback_json",
    )
    retry_eligible_finish_reasons: tuple[Literal["stop", "length"], ...] = (
        "stop",
        "length",
    )
    outer_orchestration_policy_version: Literal[
        "portfolio-s1-feedback-round3-global-retry-v2"
    ] = QWEN38_FEEDBACK_ROUND3_RETRY_POLICY_VERSION_V2
    outer_orchestration_policy_sha256: Literal[
        QWEN38_FEEDBACK_ROUND3_RETRY_POLICY_SHA256_V2
    ] = QWEN38_FEEDBACK_ROUND3_RETRY_POLICY_SHA256_V2
    attempt_transport_policy_version: Literal[
        "visual-feedback-qwen38-dashscope-json-schema-round3-primary-v1"
    ] = ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_VERSION_V1
    attempt_transport_policy_sha256: Literal[
        ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    ] = ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
    attempt_response_format: Literal["json_schema"] = "json_schema"
    requested_json_schema_policy_version: Literal[
        "visual-feedback-output-json-schema-v1"
    ] = QWEN38_FEEDBACK_JSON_SCHEMA_POLICY_VERSION
    requested_json_schema_name: Literal["visual_feedback_output_v1"] = (
        QWEN38_FEEDBACK_JSON_SCHEMA_NAME
    )
    requested_json_schema_sha256: Literal[QWEN38_FEEDBACK_JSON_SCHEMA_SHA256] = (
        QWEN38_FEEDBACK_JSON_SCHEMA_SHA256
    )
    requested_json_schema_strict: Literal[True] = True
    attempt_transport_retry_policy: Literal[
        "no_internal_retry_each_provider_attempt"
    ] = "no_internal_retry_each_provider_attempt"
    maximum_reservation_cny: Literal["6.923160000000"] = (
        QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_FRESH_RESERVATION_CNY
    )
    prior_cumulative_actual_cost_cny: Literal["24.319500000000"] = (
        QWEN38_FEEDBACK_ROUND3_SCHEMA_PRIOR_ACTUAL_COST_CNY
    )
    cumulative_maximum_reservation_cny: Literal["31.242660000000"] = (
        QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_CUMULATIVE_MAXIMUM_CNY
    )
    technical_phase_hard_cap_cny: Literal["10.000000000000"] = (
        QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_FRESH_STAGE_HARD_CAP_CNY
    )
    live_cumulative_hard_cap_cny: Literal["34.319500000000"] = (
        QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_CUMULATIVE_HARD_CAP_CNY
    )
    future_full_fresh_maximum_reservation_cny: Literal["112.155192000000"] = (
        QWEN38_FEEDBACK_MAXIMUM_RESERVATION_CNY_V3
    )
    future_full_cumulative_maximum_reservation_cny: Literal["136.474692000000"] = (
        QWEN38_FEEDBACK_ROUND3_SCHEMA_CUMULATIVE_MAXIMUM_CNY
    )
    future_full_technical_cumulative_hard_cap_cny: Literal["137.000000000000"] = (
        QWEN38_FEEDBACK_ROUND3_SCHEMA_CUMULATIVE_TECHNICAL_HARD_CAP_CNY
    )
    future_full_envelope_live_authorized: Literal[False] = False
    future_full_owner_budget_authorization_status: Literal["not_granted"] = (
        "not_granted"
    )
    phase60_requires_new_owner_approval: Literal[True] = True
    full_run_and_retry_scope_owner_approved: Literal[False] = False
    live_call_authority_scope: Literal["canary12_plus_three_global_retries_only"] = (
        "canary12_plus_three_global_retries_only"
    )
    technical_cap_scope: Literal["prior_actual_plus_fresh_reservations"] = (
        "prior_actual_plus_fresh_reservations"
    )
    fresh_run_and_retry_scope_owner_approved: Literal[True] = True
    live_provider_calls_authorized: Literal[True] = True
    live_use_requires_separate_owner_budget_authorization: Literal[False] = False
    owner_budget_authorization_scope: Literal[
        "round3-json-schema-canary12-plus-three-global-same-entry-parser-or-"
        "length-retries"
    ] = (
        "round3-json-schema-canary12-plus-three-global-same-entry-parser-or-"
        "length-retries"
    )
    owner_budget_authorization_status: Literal[
        "granted_round3_json_schema_canary12_plus_three_retries_under_cny10_"
        "fresh_stage_cap"
    ] = (
        "granted_round3_json_schema_canary12_plus_three_retries_under_cny10_"
        "fresh_stage_cap"
    )
    owner_budget_authorized_cap_cny: Literal["10.000000000000"] = (
        QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_FRESH_STAGE_HARD_CAP_CNY
    )
    owner_budget_authorized_on: Literal["2026-08-11"] = "2026-08-11"
    historical_feedback_outputs_imported: Literal[0] = 0
    terminated_json_object_canary_outputs_imported: Literal[0] = 0
    prior_actual_includes_terminated_json_object_canary: Literal[True] = True

    @field_validator(
        "phase_counts",
        "live_authorized_phase_counts",
        "retry_eligible_error_codes",
        "retry_eligible_finish_reasons",
        mode="before",
    )
    @classmethod
    def _round3_schema_tuples(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_v7_lock(self) -> Self:
        if (
            self.phase_counts != (12, 60, 120, 240)
            or self.live_authorized_phase_counts != (12,)
            or self.concurrency != 2
            or self.retry_eligible_error_codes != ("invalid_feedback_json",)
            or self.retry_eligible_finish_reasons != ("stop", "length")
            or self.attempt_response_format != "json_schema"
            or not self.requested_json_schema_strict
            or Decimal(self.per_call_reservation_cny) * self.provider_call_ceiling
            != Decimal(self.maximum_reservation_cny)
            or Decimal(self.maximum_reservation_cny)
            >= Decimal(self.technical_phase_hard_cap_cny)
            or Decimal(self.prior_cumulative_actual_cost_cny)
            + Decimal(self.maximum_reservation_cny)
            != Decimal(self.cumulative_maximum_reservation_cny)
            or Decimal(self.cumulative_maximum_reservation_cny)
            >= Decimal(self.live_cumulative_hard_cap_cny)
            or Decimal(self.prior_cumulative_actual_cost_cny)
            + Decimal(self.technical_phase_hard_cap_cny)
            != Decimal(self.live_cumulative_hard_cap_cny)
            or Decimal(self.future_full_fresh_maximum_reservation_cny)
            + Decimal(self.prior_cumulative_actual_cost_cny)
            != Decimal(self.future_full_cumulative_maximum_reservation_cny)
            or Decimal(self.future_full_cumulative_maximum_reservation_cny)
            >= Decimal(self.future_full_technical_cumulative_hard_cap_cny)
            or self.future_full_envelope_live_authorized
            or self.future_full_owner_budget_authorization_status != "not_granted"
            or self.full_run_and_retry_scope_owner_approved
            or not self.phase60_requires_new_owner_approval
            or self.terminated_json_object_canary_outputs_imported != 0
            or not self.prior_actual_includes_terminated_json_object_canary
            or not self.live_provider_calls_authorized
        ):
            raise ValueError("Qwen3.8 Feedback pricing v7 envelope drifted")
        if self.pricing_lock_sha256 != _self_hash(self, "pricing_lock_sha256"):
            raise ValueError("Qwen3.8 Feedback pricing lock v7 self hash mismatch")
        return self


class Qwen38FeedbackPricingLockV8(Qwen38FeedbackPricingLockV7):
    """Owner-approved live phase60 reservation over the completed prefix."""

    schema_version: Literal[8] = 8
    policy_version: Literal["portfolio-s1-qwen38-feedback-pricing-lock-v8"] = (
        QWEN38_FEEDBACK_PRICING_LOCK_POLICY_VERSION_V8
    )
    provider_call_ceiling: Literal[60] = (
        QWEN38_FEEDBACK_PHASE60_NEW_PROVIDER_CALL_CEILING
    )
    cumulative_provider_call_ceiling: Literal[75] = (
        QWEN38_FEEDBACK_PHASE60_CUMULATIVE_PROVIDER_CALL_CEILING
    )
    live_authorized_selected_query_count: Literal[60] = 60
    planned_phase_selected_query_count: Literal[60] = 60
    live_authorized_phase_counts: tuple[int, ...] = (60,)
    phase60_prefix_selected_count: Literal[12] = (
        QWEN38_FEEDBACK_PHASE60_PREFIX_SELECTED_COUNT
    )
    phase60_new_first_call_count: Literal[48] = (
        QWEN38_FEEDBACK_PHASE60_NEW_FIRST_CALL_COUNT
    )
    global_retry_token_count: Literal[12] = (
        QWEN38_FEEDBACK_PHASE60_NEW_RETRY_TOKEN_COUNT
    )
    prefix_global_retry_token_count: Literal[3] = (
        QWEN38_FEEDBACK_PHASE60_PREFIX_RETRY_COUNT
    )
    prefix_global_retry_tokens_reusable: Literal[False] = False
    retry_policy: Literal[
        "twelve_global_same_entry_json_schema_parser_or_length_retries_phase60_v1"
    ] = "twelve_global_same_entry_json_schema_parser_or_length_retries_phase60_v1"
    outer_orchestration_policy_version: Literal[
        "portfolio-s1-feedback-round3-phase60-global-retry-v1"
    ] = ROUND3_PHASE60_RETRY_POLICY_VERSION_V1
    outer_orchestration_policy_sha256: Literal[
        ROUND3_PHASE60_RETRY_POLICY_SHA256_V1
    ] = ROUND3_PHASE60_RETRY_POLICY_SHA256_V1
    maximum_reservation_cny: Literal["27.692640000000"] = (
        QWEN38_FEEDBACK_PHASE60_FRESH_MAXIMUM_RESERVATION_CNY
    )
    prior_cumulative_actual_cost_cny: Literal["26.264100000000"] = (
        QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_CUMULATIVE_ACTUAL_CNY
    )
    canary_prefix_actual_cost_cny: Literal["1.944600000000"] = (
        QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_ACTUAL_COST_CNY
    )
    cumulative_maximum_reservation_cny: Literal["53.956740000000"] = (
        QWEN38_FEEDBACK_PHASE60_CUMULATIVE_MAXIMUM_CNY
    )
    technical_phase_hard_cap_cny: Literal["28.000000000000"] = (
        QWEN38_FEEDBACK_PHASE60_FRESH_TECHNICAL_HARD_CAP_CNY
    )
    live_cumulative_hard_cap_cny: Literal["54.264100000000"] = (
        QWEN38_FEEDBACK_PHASE60_CUMULATIVE_TECHNICAL_HARD_CAP_CNY
    )
    future_full_cumulative_maximum_reservation_cny: Literal["138.419292000000"] = (
        "138.419292000000"
    )
    future_full_technical_cumulative_hard_cap_cny: Literal["139.000000000000"] = (
        "139.000000000000"
    )
    fresh_run_and_retry_scope_owner_approved: Literal[True] = True
    live_provider_calls_authorized: Literal[True] = True
    live_use_requires_separate_owner_budget_authorization: Literal[False] = False
    live_call_authority_scope: Literal[
        "phase60_prefix12_plus_48_first_plus_up_to_12_retries_only"
    ] = "phase60_prefix12_plus_48_first_plus_up_to_12_retries_only"
    owner_budget_authorization_decision_source: Literal["current_user_instruction"] = (
        "current_user_instruction"
    )
    owner_budget_authorization_scope: Literal[
        "phase60-prefix12-plus-48-new-firsts-plus-up-to-12-new-global-retries"
    ] = "phase60-prefix12-plus-48-new-firsts-plus-up-to-12-new-global-retries"
    owner_budget_authorization_status: Literal[
        "granted_phase60_budget_and_retry_approval"
    ] = "granted_phase60_budget_and_retry_approval"
    owner_budget_authorized_cap_cny: Literal["28.000000000000"] = (
        QWEN38_FEEDBACK_PHASE60_FRESH_TECHNICAL_HARD_CAP_CNY
    )
    owner_budget_authorized_on: Literal["2026-08-11"] = "2026-08-11"
    planned_owner_budget_cap_cny: Literal["28.000000000000"] = (
        QWEN38_FEEDBACK_PHASE60_FRESH_TECHNICAL_HARD_CAP_CNY
    )
    owner_phase60_retry_authorization_status: Literal["granted"] = "granted"
    future_full_envelope_live_authorized: Literal[False] = False
    future_full_owner_budget_authorization_status: Literal["not_granted"] = (
        "not_granted"
    )
    full_run_and_retry_scope_owner_approved: Literal[False] = False
    phase60_requires_new_owner_approval: Literal[False] = False
    phase120_requires_new_owner_approval: Literal[True] = True
    historical_feedback_outputs_imported: Literal[12] = 12
    terminated_json_object_canary_outputs_imported: Literal[0] = 0
    prior_actual_includes_terminated_json_object_canary: Literal[True] = True
    prior_actual_includes_completed_schema_canary: Literal[True] = True
    canary_prefix_run_file_sha256: Literal[
        QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_RUN_FILE_SHA256
    ] = QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_RUN_FILE_SHA256
    canary_prefix_run_sha256: Literal[
        QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_RUN_SHA256
    ] = QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_RUN_SHA256
    canary_prefix_artifact_set_sha256: Literal[
        QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_ARTIFACT_SET_SHA256
    ] = QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_ARTIFACT_SET_SHA256
    canary_prefix_selection_sha256: Literal[
        QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_SELECTION_SHA256
    ] = QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_SELECTION_SHA256

    @model_validator(mode="after")
    def _validate_v7_lock(self) -> Self:
        if (
            self.phase_counts != (12, 60, 120, 240)
            or self.live_authorized_phase_counts != (60,)
            or self.live_authorized_selected_query_count != 60
            or self.concurrency != 2
            or self.retry_eligible_error_codes != ("invalid_feedback_json",)
            or self.retry_eligible_finish_reasons != ("stop", "length")
            or self.attempt_response_format != "json_schema"
            or not self.requested_json_schema_strict
            or self.phase60_prefix_selected_count + self.phase60_new_first_call_count
            != self.planned_phase_selected_query_count
            or self.phase60_new_first_call_count + self.global_retry_token_count
            != self.provider_call_ceiling
            or QWEN38_FEEDBACK_PHASE60_PREFIX_PROVIDER_CALL_COUNT
            + self.provider_call_ceiling
            != self.cumulative_provider_call_ceiling
            or self.prefix_global_retry_tokens_reusable
            or Decimal(self.per_call_reservation_cny) * self.provider_call_ceiling
            != Decimal(self.maximum_reservation_cny)
            or Decimal(self.maximum_reservation_cny)
            >= Decimal(self.technical_phase_hard_cap_cny)
            or Decimal(self.prior_cumulative_actual_cost_cny)
            + Decimal(self.maximum_reservation_cny)
            != Decimal(self.cumulative_maximum_reservation_cny)
            or Decimal(self.prior_cumulative_actual_cost_cny)
            + Decimal(self.technical_phase_hard_cap_cny)
            != Decimal(self.live_cumulative_hard_cap_cny)
            or Decimal(self.prior_cumulative_actual_cost_cny)
            != Decimal(QWEN38_FEEDBACK_ROUND3_SCHEMA_PRIOR_ACTUAL_COST_CNY)
            + Decimal(self.canary_prefix_actual_cost_cny)
            or Decimal(self.future_full_fresh_maximum_reservation_cny)
            + Decimal(self.prior_cumulative_actual_cost_cny)
            != Decimal(self.future_full_cumulative_maximum_reservation_cny)
            or Decimal(self.future_full_cumulative_maximum_reservation_cny)
            >= Decimal(self.future_full_technical_cumulative_hard_cap_cny)
            or not self.fresh_run_and_retry_scope_owner_approved
            or not self.live_provider_calls_authorized
            or self.live_use_requires_separate_owner_budget_authorization
            or self.owner_budget_authorization_status
            != "granted_phase60_budget_and_retry_approval"
            or self.owner_budget_authorized_cap_cny
            != QWEN38_FEEDBACK_PHASE60_FRESH_TECHNICAL_HARD_CAP_CNY
            or self.owner_budget_authorized_on != "2026-08-11"
            or self.owner_phase60_retry_authorization_status != "granted"
            or self.future_full_envelope_live_authorized
            or self.full_run_and_retry_scope_owner_approved
            or self.phase60_requires_new_owner_approval
            or not self.phase120_requires_new_owner_approval
            or self.historical_feedback_outputs_imported != 12
            or self.terminated_json_object_canary_outputs_imported != 0
            or not self.prior_actual_includes_completed_schema_canary
        ):
            raise ValueError("Qwen3.8 Feedback pricing v8 live envelope drifted")
        if self.pricing_lock_sha256 != _self_hash(self, "pricing_lock_sha256"):
            raise ValueError("Qwen3.8 Feedback pricing lock v8 self hash mismatch")
        return self


class Qwen38FeedbackRoleSelectionV12(_StrictFrozenModel):
    """Typed active-role lock for the live fresh-v3 Feedback plan."""

    schema_version: Literal[12] = 12
    artifact_kind: Literal["model_role_selection"] = "model_role_selection"
    assistant: dict[str, object]
    author: dict[str, object]
    authorization_status: dict[str, object]
    evaluator_isolation: dict[str, object]
    feedback_evaluator: dict[str, object]
    formal_status: dict[str, object]
    label_synthesis: dict[str, object]
    offline_judge: dict[str, object]
    owner_decision: str
    selected_on: Literal["2026-08-10"] = "2026-08-10"
    status: Literal[
        "owner_selected_portfolio_qwen38_feedback_fresh_v3_live_authorized"
    ] = "owner_selected_portfolio_qwen38_feedback_fresh_v3_live_authorized"
    supersedes_for_active_portfolio: Literal[
        "specs/authoring/model-role-selection-v11.json"
    ] = "specs/authoring/model-role-selection-v11.json"
    selection_sha256: Sha256

    @model_validator(mode="after")
    def _validate_selection(self) -> Self:
        expected_feedback = {
            "provider": QWEN38_FEEDBACK_PROVIDER,
            "model": QWEN38_FEEDBACK_MODEL,
            "cache_namespace": QWEN38_FEEDBACK_CACHE_NAMESPACE_V2,
            "result_schema_version": QWEN38_FEEDBACK_RESULT_SCHEMA_VERSION_V2,
            "bound_artifact_schema_version": (
                QWEN38_FEEDBACK_BOUND_ARTIFACT_SCHEMA_VERSION_V2
            ),
            "bound_artifact_policy_version": (
                QWEN38_FEEDBACK_BOUND_ARTIFACT_POLICY_VERSION_V2
            ),
            "enable_thinking": True,
            "thinking_budget": QWEN38_FEEDBACK_THINKING_BUDGET,
            "max_tokens": None,
            "max_completion_tokens": QWEN38_FEEDBACK_MAX_COMPLETION_TOKENS_V2,
            "max_completion_tokens_documented_upper_tolerance_tokens": (
                QWEN38_FEEDBACK_MAX_COMPLETION_TOKENS_TOLERANCE
            ),
            "output_token_reservation_ceiling_per_call": (
                QWEN38_FEEDBACK_OUTPUT_RESERVATION_TOKENS_V2
            ),
            "per_call_reservation_cny": (QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY_V2),
            "provider_call_ceiling": QWEN38_FEEDBACK_PROVIDER_CALL_CEILING_V3,
            "worst_case_reservation_cny": (QWEN38_FEEDBACK_MAXIMUM_RESERVATION_CNY_V3),
            "technical_phase_hard_cap_cny": (
                QWEN38_FEEDBACK_TECHNICAL_PHASE_HARD_CAP_CNY_V2
            ),
            "owner_authorized_budget_ceiling_cny": (
                QWEN38_FEEDBACK_OWNER_AUTHORIZED_BUDGET_CEILING_CNY
            ),
            "normal_attempts_per_selected_query": 1,
            "global_retry_token_count": 3,
            "max_attempts_per_retried_query": 2,
            "retry_policy": ("three_global_same_entry_schema_or_length_retries_v2"),
            "retry_eligible_error_codes": ["invalid_feedback_json"],
            "retry_eligible_finish_reasons": ["stop", "length"],
            "outer_orchestration_policy_version": (
                QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_VERSION_V2
            ),
            "outer_orchestration_policy_sha256": (
                QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V2
            ),
            "attempt_transport_policy_version": (
                QWEN38_FEEDBACK_TRANSPORT_POLICY_VERSION_V7
            ),
            "attempt_transport_policy_sha256": (
                QWEN38_FEEDBACK_TRANSPORT_POLICY_SHA256_V7
            ),
            "transport_policy_version": (QWEN38_FEEDBACK_TRANSPORT_POLICY_VERSION_V7),
            "transport_policy_sha256": QWEN38_FEEDBACK_TRANSPORT_POLICY_SHA256_V7,
            "model_source_lock_file_sha256": (
                QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V2
            ),
            "model_source_lock_sha256": QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V2,
            "pricing_lock_file_sha256": (QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V5),
            "pricing_lock_sha256": QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V5,
            "control_policy_version": "portfolio-s1-feedback-control-v12",
            "launch_policy_version": ("portfolio-s1-qwen38-feedback-launch-lock-v5"),
            "fresh_run_and_retry_scope_owner_approved": True,
            "live_provider_calls_authorized": True,
            "budget_authorization_decision_source": "current_user_instruction",
            "budget_authorization_status": (
                "fresh_v3_live_authorized_cny150_owner_ceiling_cny113_technical_stop"
            ),
            "budget_authorized_on": "2026-08-10",
        }
        if any(
            self.feedback_evaluator.get(key) != value
            for key, value in expected_feedback.items()
        ):
            raise ValueError("Qwen3.8 Feedback role v12 runtime identity drifted")
        if (
            self.authorization_status.get(
                "historical_authorizations_reusable_for_new_roles"
            )
            is not False
            or self.evaluator_isolation.get("cache_namespaces")
            != [QWEN38_FEEDBACK_CACHE_NAMESPACE_V2, "final-evaluator-v11"]
            or self.selection_sha256 != _self_hash(self, "selection_sha256")
        ):
            raise ValueError("Qwen3.8 Feedback role v12 governance drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class Qwen38FeedbackRoleSelectionV13(_StrictFrozenModel):
    """Typed active-role lock for wholly fresh Round3 JSON-object Feedback."""

    schema_version: Literal[13] = 13
    artifact_kind: Literal["model_role_selection"] = "model_role_selection"
    assistant: dict[str, object]
    author: dict[str, object]
    authorization_status: dict[str, object]
    evaluator_isolation: dict[str, object]
    feedback_evaluator: dict[str, object]
    formal_status: dict[str, object]
    label_synthesis: dict[str, object]
    offline_judge: dict[str, object]
    owner_decision: str
    selected_on: Literal["2026-08-11"] = "2026-08-11"
    status: Literal[
        "owner_selected_portfolio_qwen38_feedback_round3_json_object_live_authorized"
    ] = "owner_selected_portfolio_qwen38_feedback_round3_json_object_live_authorized"
    supersedes_for_active_portfolio: Literal[
        "specs/authoring/model-role-selection-v12.json"
    ] = "specs/authoring/model-role-selection-v12.json"
    selection_sha256: Sha256

    @model_validator(mode="after")
    def _validate_selection(self) -> Self:
        expected_feedback = {
            "provider": QWEN38_FEEDBACK_PROVIDER,
            "model": QWEN38_FEEDBACK_MODEL,
            "model_revision": "moving-alias",
            "model_frozen_snapshot": False,
            "processor": QWEN38_FEEDBACK_PROCESSOR,
            "cache_namespace": QWEN38_FEEDBACK_CACHE_NAMESPACE_V3,
            "result_schema_version": QWEN38_FEEDBACK_RESULT_SCHEMA_VERSION_V3,
            "bound_artifact_schema_version": (
                QWEN38_FEEDBACK_BOUND_ARTIFACT_SCHEMA_VERSION_V3
            ),
            "bound_artifact_policy_version": (
                QWEN38_FEEDBACK_BOUND_ARTIFACT_POLICY_VERSION_V3
            ),
            "wire_kind": QWEN38_FEEDBACK_ROUND3_WIRE_KIND_V1,
            "requested_response_format": "json_object",
            "requested_json_schema": False,
            "requested_json_schema_strict": False,
            "enable_thinking": True,
            "thinking_budget": QWEN38_FEEDBACK_THINKING_BUDGET,
            "max_tokens": None,
            "max_completion_tokens": QWEN38_FEEDBACK_MAX_COMPLETION_TOKENS_V2,
            "max_completion_tokens_documented_upper_tolerance_tokens": (
                QWEN38_FEEDBACK_MAX_COMPLETION_TOKENS_TOLERANCE
            ),
            "requested_stream": False,
            "temperature": None,
            "top_p": None,
            "seed": None,
            "timeout_seconds": QWEN38_FEEDBACK_TIMEOUT_SECONDS,
            "provider_internal_max_attempts": 1,
            "selected_query_count": QWEN38_FEEDBACK_SELECTED_COUNT,
            "phase_counts": [12, 60, 120, 240],
            "normal_attempts_per_selected_query": 1,
            "global_retry_token_count": 3,
            "max_lifetime_attempts_per_retried_query": 2,
            "provider_call_ceiling": QWEN38_FEEDBACK_PROVIDER_CALL_CEILING_V3,
            "retry_policy": (
                "three_global_same_entry_json_object_parser_or_length_retries_round3_v1"
            ),
            "retry_eligible_error_codes": ["invalid_feedback_json"],
            "retry_eligible_finish_reasons": ["stop", "length"],
            "outer_orchestration_policy_version": (
                QWEN38_FEEDBACK_ROUND3_RETRY_POLICY_VERSION_V1
            ),
            "outer_orchestration_policy_sha256": (
                QWEN38_FEEDBACK_ROUND3_RETRY_POLICY_SHA256_V1
            ),
            "attempt_transport_policy_version": (
                ROUND3_PRIMARY_TRANSPORT_POLICY_VERSION_V1
            ),
            "attempt_transport_policy_sha256": (
                ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1
            ),
            "transport_policy_version": ROUND3_PRIMARY_TRANSPORT_POLICY_VERSION_V1,
            "transport_policy_sha256": ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1,
            "prompt_policy_version": (QWEN38_FEEDBACK_ROUND3_PROMPT_POLICY_VERSION_V6),
            "prompt_policy_sha256": QWEN38_FEEDBACK_ROUND3_PROMPT_POLICY_SHA256_V6,
            "parser_policy_version": (QWEN38_FEEDBACK_ROUND3_PARSER_POLICY_VERSION_V3),
            "parser_policy_sha256": QWEN38_FEEDBACK_ROUND3_PARSER_POLICY_SHA256_V3,
            "model_source_lock_file": QWEN38_FEEDBACK_SOURCE_LOCK_RELATIVE_PATH_V3,
            "model_source_lock_file_sha256": (
                QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V3
            ),
            "model_source_lock_sha256": QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V3,
            "pricing_lock_file": QWEN38_FEEDBACK_PRICING_LOCK_RELATIVE_PATH_V6,
            "pricing_lock_file_sha256": (QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V6),
            "pricing_lock_sha256": QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V6,
            "input_token_reservation_ceiling_per_call": (
                QWEN38_FEEDBACK_INPUT_RESERVATION_TOKENS
            ),
            "output_token_reservation_ceiling_per_call": (
                QWEN38_FEEDBACK_OUTPUT_RESERVATION_TOKENS_V2
            ),
            "per_call_reservation_cny": QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY_V2,
            "fresh_maximum_reservation_cny": QWEN38_FEEDBACK_MAXIMUM_RESERVATION_CNY_V3,
            "prior_cumulative_actual_cost_cny": (
                QWEN38_FEEDBACK_ROUND3_PRIOR_ACTUAL_COST_CNY
            ),
            "cumulative_maximum_reservation_cny": (
                QWEN38_FEEDBACK_ROUND3_CUMULATIVE_MAXIMUM_CNY
            ),
            "technical_cumulative_hard_cap_cny": (
                QWEN38_FEEDBACK_ROUND3_CUMULATIVE_TECHNICAL_HARD_CAP_CNY
            ),
            "owner_authorized_budget_ceiling_cny": (
                QWEN38_FEEDBACK_OWNER_AUTHORIZED_BUDGET_CEILING_CNY
            ),
            "control_policy_version": "portfolio-s1-feedback-round3-control-v1",
            "launch_policy_version": "portfolio-s1-feedback-round3-launch-v1",
            "historical_feedback_outputs_imported": 0,
            "image_source_membership_ancestry_only": True,
            "old_remote_auth_does_not_authorize_round3_transport": True,
            "live_call_authority": True,
            "live_provider_calls_authorized": True,
            "budget_authorization_decision_source": "current_user_instruction",
            "budget_authorization_status": (
                "round3_json_object_live_authorized_cny150_owner_ceiling_cny135_"
                "cumulative_technical_stop"
            ),
            "budget_authorized_on": "2026-08-11",
        }
        if any(
            self.feedback_evaluator.get(key) != value
            for key, value in expected_feedback.items()
        ):
            raise ValueError("Qwen3.8 Feedback role v13 runtime identity drifted")
        if (
            self.feedback_evaluator.get("requested_json_schema_name") is not None
            or self.feedback_evaluator.get("requested_json_schema_sha256") is not None
            or self.authorization_status.get(
                "historical_authorizations_reusable_for_new_roles"
            )
            is not False
            or self.authorization_status.get(
                "old_remote_auth_does_not_authorize_round3_transport"
            )
            is not True
            or self.evaluator_isolation.get("cache_namespaces")
            != [QWEN38_FEEDBACK_CACHE_NAMESPACE_V3, "final-evaluator-v11"]
            or self.selection_sha256 != _self_hash(self, "selection_sha256")
        ):
            raise ValueError("Qwen3.8 Feedback role v13 governance drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class Qwen38FeedbackRoleSelectionV14(_StrictFrozenModel):
    """Typed role lock for the authorized strict-schema Round3 canary."""

    schema_version: Literal[14] = 14
    artifact_kind: Literal["model_role_selection"] = "model_role_selection"
    assistant: dict[str, object]
    author: dict[str, object]
    authorization_status: dict[str, object]
    evaluator_isolation: dict[str, object]
    feedback_evaluator: dict[str, object]
    formal_status: dict[str, object]
    label_synthesis: dict[str, object]
    offline_judge: dict[str, object]
    owner_decision: str
    selected_on: Literal["2026-08-11"] = "2026-08-11"
    status: Literal[
        "owner_selected_portfolio_qwen38_feedback_round3_json_schema_canary_authorized"
    ] = "owner_selected_portfolio_qwen38_feedback_round3_json_schema_canary_authorized"
    supersedes_for_active_portfolio: Literal[
        "specs/authoring/model-role-selection-v13.json"
    ] = "specs/authoring/model-role-selection-v13.json"
    selection_sha256: Sha256

    @model_validator(mode="after")
    def _validate_selection(self) -> Self:
        expected_feedback = {
            "provider": QWEN38_FEEDBACK_PROVIDER,
            "model": QWEN38_FEEDBACK_MODEL,
            "model_revision": "moving-alias",
            "model_frozen_snapshot": False,
            "processor": QWEN38_FEEDBACK_PROCESSOR,
            "cache_namespace": QWEN38_FEEDBACK_CACHE_NAMESPACE_V4,
            "result_schema_version": QWEN38_FEEDBACK_RESULT_SCHEMA_VERSION_V4,
            "bound_artifact_schema_version": (
                QWEN38_FEEDBACK_BOUND_ARTIFACT_SCHEMA_VERSION_V4
            ),
            "bound_artifact_policy_version": (
                QWEN38_FEEDBACK_BOUND_ARTIFACT_POLICY_VERSION_V4
            ),
            "wire_kind": QWEN38_FEEDBACK_ROUND3_WIRE_KIND_V2,
            "requested_response_format": "json_schema",
            "requested_json_schema": True,
            "requested_json_schema_policy_version": (
                QWEN38_FEEDBACK_JSON_SCHEMA_POLICY_VERSION
            ),
            "requested_json_schema_name": QWEN38_FEEDBACK_JSON_SCHEMA_NAME,
            "requested_json_schema_sha256": QWEN38_FEEDBACK_JSON_SCHEMA_SHA256,
            "requested_json_schema_strict": True,
            "enable_thinking": True,
            "thinking_budget": QWEN38_FEEDBACK_THINKING_BUDGET,
            "max_tokens": None,
            "max_completion_tokens": QWEN38_FEEDBACK_MAX_COMPLETION_TOKENS_V2,
            "max_completion_tokens_documented_upper_tolerance_tokens": (
                QWEN38_FEEDBACK_MAX_COMPLETION_TOKENS_TOLERANCE
            ),
            "requested_stream": False,
            "temperature": None,
            "top_p": None,
            "seed": None,
            "timeout_seconds": QWEN38_FEEDBACK_TIMEOUT_SECONDS,
            "provider_internal_max_attempts": 1,
            "selected_query_count": QWEN38_FEEDBACK_SELECTED_COUNT,
            "live_authorized_selected_query_count": (
                QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_SELECTED_COUNT
            ),
            "phase_counts": [12, 60, 120, 240],
            "live_authorized_phase_counts": [12],
            "concurrency": 2,
            "normal_attempts_per_selected_query": 1,
            "global_retry_token_count": 3,
            "max_lifetime_attempts_per_retried_query": 2,
            "provider_call_ceiling": (
                QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_PROVIDER_CALL_CEILING
            ),
            "future_full_provider_call_ceiling": (
                QWEN38_FEEDBACK_PROVIDER_CALL_CEILING_V3
            ),
            "retry_policy": (
                "three_global_same_entry_json_schema_parser_or_length_retries_round3_v2"
            ),
            "retry_eligible_error_codes": ["invalid_feedback_json"],
            "retry_eligible_finish_reasons": ["stop", "length"],
            "outer_orchestration_policy_version": (
                QWEN38_FEEDBACK_ROUND3_RETRY_POLICY_VERSION_V2
            ),
            "outer_orchestration_policy_sha256": (
                QWEN38_FEEDBACK_ROUND3_RETRY_POLICY_SHA256_V2
            ),
            "attempt_transport_policy_version": (
                ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_VERSION_V1
            ),
            "attempt_transport_policy_sha256": (
                ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
            ),
            "attempt_response_format": "json_schema",
            "attempt_transport_retry_policy": (
                "no_internal_retry_each_provider_attempt"
            ),
            "transport_policy_version": (
                ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_VERSION_V1
            ),
            "transport_policy_sha256": (
                ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
            ),
            "prompt_policy_version": (QWEN38_FEEDBACK_ROUND3_PROMPT_POLICY_VERSION_V6),
            "prompt_policy_sha256": QWEN38_FEEDBACK_ROUND3_PROMPT_POLICY_SHA256_V6,
            "parser_policy_version": (QWEN38_FEEDBACK_ROUND3_PARSER_POLICY_VERSION_V3),
            "parser_policy_sha256": QWEN38_FEEDBACK_ROUND3_PARSER_POLICY_SHA256_V3,
            "model_source_lock_file": QWEN38_FEEDBACK_SOURCE_LOCK_RELATIVE_PATH_V4,
            "model_source_lock_file_sha256": (
                QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V4
            ),
            "model_source_lock_sha256": QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V4,
            "pricing_lock_file": QWEN38_FEEDBACK_PRICING_LOCK_RELATIVE_PATH_V7,
            "pricing_lock_file_sha256": (QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V7),
            "pricing_lock_sha256": QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V7,
            "input_token_reservation_ceiling_per_call": (
                QWEN38_FEEDBACK_INPUT_RESERVATION_TOKENS
            ),
            "output_token_reservation_ceiling_per_call": (
                QWEN38_FEEDBACK_OUTPUT_RESERVATION_TOKENS_V2
            ),
            "per_call_reservation_cny": QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY_V2,
            "fresh_maximum_reservation_cny": (
                QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_FRESH_RESERVATION_CNY
            ),
            "prior_cumulative_actual_cost_cny": (
                QWEN38_FEEDBACK_ROUND3_SCHEMA_PRIOR_ACTUAL_COST_CNY
            ),
            "cumulative_maximum_reservation_cny": (
                QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_CUMULATIVE_MAXIMUM_CNY
            ),
            "fresh_stage_hard_cap_cny": (
                QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_FRESH_STAGE_HARD_CAP_CNY
            ),
            "live_cumulative_hard_cap_cny": (
                QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_CUMULATIVE_HARD_CAP_CNY
            ),
            "future_full_fresh_maximum_reservation_cny": (
                QWEN38_FEEDBACK_MAXIMUM_RESERVATION_CNY_V3
            ),
            "future_full_cumulative_maximum_reservation_cny": (
                QWEN38_FEEDBACK_ROUND3_SCHEMA_CUMULATIVE_MAXIMUM_CNY
            ),
            "future_full_technical_cumulative_hard_cap_cny": (
                QWEN38_FEEDBACK_ROUND3_SCHEMA_CUMULATIVE_TECHNICAL_HARD_CAP_CNY
            ),
            "owner_authorized_budget_ceiling_cny": (
                QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_FRESH_STAGE_HARD_CAP_CNY
            ),
            "live_call_authority_scope": ("canary12_plus_three_global_retries_only"),
            "future_full_envelope_live_authorized": False,
            "future_full_owner_budget_authorization_status": "not_granted",
            "full_run_and_retry_scope_owner_approved": False,
            "phase60_requires_new_owner_approval": True,
            "control_policy_version": "portfolio-s1-feedback-round3-control-v2",
            "launch_policy_version": "portfolio-s1-feedback-round3-launch-v2",
            "historical_feedback_outputs_imported": 0,
            "terminated_json_object_canary_outputs_imported": 0,
            "prior_actual_includes_terminated_json_object_canary": True,
            "image_source_membership_ancestry_only": True,
            "old_remote_auth_does_not_authorize_round3_transport": True,
            "live_call_authority": True,
            "live_provider_calls_authorized": True,
            "budget_authorization_decision_source": "current_user_instruction",
            "budget_authorization_status": (
                "round3_json_schema_canary12_plus_three_retries_authorized_"
                "under_cny10_fresh_stage_cap"
            ),
            "budget_authorized_on": "2026-08-11",
        }
        if any(
            self.feedback_evaluator.get(key) != value
            for key, value in expected_feedback.items()
        ):
            raise ValueError("Qwen3.8 Feedback role v14 runtime identity drifted")
        if (
            self.authorization_status.get(
                "historical_authorizations_reusable_for_new_roles"
            )
            is not False
            or self.authorization_status.get(
                "old_remote_auth_does_not_authorize_round3_transport"
            )
            is not True
            or self.authorization_status.get("dashscope_qwen38_feedback")
            != (
                "live Round3 JSON-Schema authority is limited to canary12 plus "
                "three global same-entry retries under a fresh CNY10 stage cap; "
                "phase60 and the future full243 envelope require new owner approval"
            )
            or self.evaluator_isolation.get("cache_namespaces")
            != [QWEN38_FEEDBACK_CACHE_NAMESPACE_V4, "final-evaluator-v11"]
            or self.selection_sha256 != _self_hash(self, "selection_sha256")
        ):
            raise ValueError("Qwen3.8 Feedback role v14 governance drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class Qwen38FeedbackRoleSelectionV15(Qwen38FeedbackRoleSelectionV14):
    """Owner-selected live role for the approved phase60 envelope."""

    schema_version: Literal[15] = 15
    status: Literal[
        "owner_selected_portfolio_qwen38_feedback_phase60_live_authorized"
    ] = "owner_selected_portfolio_qwen38_feedback_phase60_live_authorized"
    supersedes_for_active_portfolio: Literal[
        "specs/authoring/model-role-selection-v14.json"
    ] = "specs/authoring/model-role-selection-v14.json"

    @model_validator(mode="after")
    def _validate_selection(self) -> Self:
        expected_feedback = {
            "provider": QWEN38_FEEDBACK_PROVIDER,
            "model": QWEN38_FEEDBACK_MODEL,
            "cache_namespace": QWEN38_FEEDBACK_CACHE_NAMESPACE_V4,
            "wire_kind": QWEN38_FEEDBACK_ROUND3_WIRE_KIND_V2,
            "requested_response_format": "json_schema",
            "requested_json_schema": True,
            "requested_json_schema_sha256": QWEN38_FEEDBACK_JSON_SCHEMA_SHA256,
            "requested_json_schema_strict": True,
            "transport_policy_version": (
                ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_VERSION_V1
            ),
            "transport_policy_sha256": (
                ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1
            ),
            "outer_orchestration_policy_version": (
                ROUND3_PHASE60_RETRY_POLICY_VERSION_V1
            ),
            "outer_orchestration_policy_sha256": (
                ROUND3_PHASE60_RETRY_POLICY_SHA256_V1
            ),
            "model_source_lock_file": QWEN38_FEEDBACK_SOURCE_LOCK_RELATIVE_PATH_V5,
            "model_source_lock_file_sha256": (
                QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V5
            ),
            "model_source_lock_sha256": QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V5,
            "pricing_lock_file": QWEN38_FEEDBACK_PRICING_LOCK_RELATIVE_PATH_V8,
            "pricing_lock_file_sha256": (QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V8),
            "pricing_lock_sha256": QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V8,
            "phase60_prefix_selected_count": 12,
            "phase60_new_first_call_count": 48,
            "new_global_retry_token_count": 12,
            "new_provider_call_ceiling": 60,
            "cumulative_provider_call_ceiling": 75,
            "prefix_global_retry_token_count": 3,
            "prefix_global_retry_tokens_reusable": False,
            "prior_cumulative_actual_cost_cny": (
                QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_CUMULATIVE_ACTUAL_CNY
            ),
            "fresh_maximum_reservation_cny": (
                QWEN38_FEEDBACK_PHASE60_FRESH_MAXIMUM_RESERVATION_CNY
            ),
            "fresh_stage_hard_cap_cny": (
                QWEN38_FEEDBACK_PHASE60_FRESH_TECHNICAL_HARD_CAP_CNY
            ),
            "cumulative_maximum_reservation_cny": (
                QWEN38_FEEDBACK_PHASE60_CUMULATIVE_MAXIMUM_CNY
            ),
            "live_cumulative_hard_cap_cny": (
                QWEN38_FEEDBACK_PHASE60_CUMULATIVE_TECHNICAL_HARD_CAP_CNY
            ),
            "canary_prefix_run_file_sha256": (
                QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_RUN_FILE_SHA256
            ),
            "canary_prefix_run_sha256": (
                QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_RUN_SHA256
            ),
            "canary_prefix_artifact_set_sha256": (
                QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_ARTIFACT_SET_SHA256
            ),
            "canary_prefix_selection_sha256": (
                QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_SELECTION_SHA256
            ),
            "historical_feedback_outputs_imported": 12,
            "terminated_json_object_canary_outputs_imported": 0,
            "live_call_authority": True,
            "live_provider_calls_authorized": True,
            "fresh_run_and_retry_scope_owner_approved": True,
            "live_authorized_selected_query_count": 60,
            "live_authorized_phase_counts": [60],
            "live_use_requires_separate_owner_budget_authorization": False,
            "live_call_authority_scope": (
                "phase60_prefix12_plus_48_first_plus_up_to_12_retries_only"
            ),
            "authorization_scope": (
                "phase60-prefix12-plus-48-new-firsts-plus-up-to-12-new-global-retries"
            ),
            "budget_authorization_decision_source": "current_user_instruction",
            "budget_authorization_status": (
                "granted_phase60_budget_and_retry_approval"
            ),
            "budget_authorized_on": "2026-08-11",
            "owner_authorized_budget_ceiling_cny": (
                QWEN38_FEEDBACK_PHASE60_FRESH_TECHNICAL_HARD_CAP_CNY
            ),
            "owner_phase60_budget_authorization_status": "granted",
            "owner_phase60_retry_authorization_status": "granted",
            "phase60_requires_new_owner_approval": False,
            "phase120_requires_new_owner_approval": True,
            "creator_authorized": False,
            "bundle_v11_publishable": False,
        }
        if any(
            self.feedback_evaluator.get(key) != value
            for key, value in expected_feedback.items()
        ):
            raise ValueError("Qwen3.8 Feedback role v15 live identity drifted")
        if (
            self.authorization_status.get(
                "historical_authorizations_reusable_for_new_roles"
            )
            is not False
            or self.authorization_status.get("dashscope_qwen38_feedback")
            != (
                "live phase60 authority is limited to the frozen canary12 prefix "
                "plus 48 new first attempts and up to twelve new global retries "
                "under a CNY27.692640 maximum reservation, CNY28 fresh technical "
                "cap, and CNY54.264100 cumulative technical cap; phase120, "
                "BundleV11, and Creator require new authority"
            )
            or self.evaluator_isolation.get("cache_namespaces")
            != [QWEN38_FEEDBACK_CACHE_NAMESPACE_V4, "final-evaluator-v11"]
            or self.selection_sha256 != _self_hash(self, "selection_sha256")
        ):
            raise ValueError("Qwen3.8 Feedback role v15 governance drifted")
        return self


class SelectedQwenFeedbackAssetV1(_StrictFrozenModel):
    query_id: str
    asset_id: str
    image_sha256: Sha256

    @field_validator("query_id", "asset_id")
    @classmethod
    def _texts(cls, value: str, info) -> str:
        return _canonical_text(value, info.field_name)


class PortfolioS1QwenFeedbackAuthorizationV3(_StrictFrozenModel):
    """Independent owner approval for exactly the frozen selected48 assets."""

    schema_version: Literal[3] = 3
    kind: Literal["portfolio-s1-feedback-selected-asset-authorization"] = (
        "portfolio-s1-feedback-selected-asset-authorization"
    )
    policy_version: Literal[
        "portfolio-s1-feedback-selected-assets-authorization-v3"
    ] = QWEN37_FEEDBACK_AUTHORIZATION_POLICY_VERSION
    authorization_id: str
    status: Literal["owner-approved"] = "owner-approved"
    scope: Literal["core-opt800-s1-feedback-selected-48-assets"] = (
        "core-opt800-s1-feedback-selected-48-assets"
    )
    reviewer_id: str
    reviewed_at: datetime
    owner_statement: str
    selection_sha256: Sha256
    provider: Literal["qwen"] = "qwen"
    model: Literal["qwen3.7-plus-2026-05-26"] = QWEN37_FEEDBACK_MODEL
    processor: Literal["dashscope-qwen37-feedback"] = QWEN37_FEEDBACK_PROCESSOR
    endpoint_configuration: Literal["DASHSCOPE_BASE_URL"] = (
        QWEN37_FEEDBACK_ENDPOINT_CONFIGURATION
    )
    requested_response_format: Literal["json_schema"] = "json_schema"
    requested_json_schema_policy_version: Literal[
        "visual-feedback-output-json-schema-v1"
    ] = QWEN37_FEEDBACK_JSON_SCHEMA_POLICY_VERSION
    requested_json_schema_name: Literal["visual_feedback_output_v1"] = (
        QWEN37_FEEDBACK_JSON_SCHEMA_NAME
    )
    requested_json_schema_sha256: Literal[
        "af673bc72a52788b4a3871b030e95123e070d23cc51388e4abb06d0d5fa9667e"
    ] = QWEN37_FEEDBACK_JSON_SCHEMA_SHA256
    requested_json_schema_strict: Literal[True] = True
    transport_policy_version: Literal[
        "visual-feedback-qwen-dashscope-json-schema-v5"
    ] = QWEN37_FEEDBACK_TRANSPORT_POLICY_VERSION
    transport_policy_sha256: Literal[
        "092f36be5eb08e4fd58edc888fe273bee2b4295059048dbd9715d472e534f20c"
    ] = QWEN37_FEEDBACK_TRANSPORT_POLICY_SHA256
    requested_stream: Literal[False] = False
    requested_thinking: Literal[True] = True
    requested_thinking_budget: Literal[2048] = QWEN37_FEEDBACK_THINKING_BUDGET
    requested_timeout_seconds: Literal[600] = QWEN37_FEEDBACK_TIMEOUT_SECONDS
    requested_temperature: None = None
    requested_top_p: None = None
    requested_seed: None = None
    requested_max_tokens: None = None
    max_completion_tokens: Literal[4096] = QWEN37_FEEDBACK_MAX_COMPLETION_TOKENS
    max_completion_tokens_documented_upper_tolerance_tokens: Literal[10] = (
        QWEN37_FEEDBACK_MAX_COMPLETION_TOKENS_TOLERANCE
    )
    selected_query_count: Literal[48] = QWEN37_FEEDBACK_SELECTED_COUNT
    provider_call_ceiling: Literal[48] = QWEN37_FEEDBACK_PROVIDER_CALL_CEILING
    max_attempts: Literal[1] = 1
    retry_policy: Literal["no_retry"] = "no_retry"
    pricing_tier_max_input_tokens: Literal[256000] = (
        QWEN37_FEEDBACK_PRICING_TIER_MAX_INPUT_TOKENS
    )
    input_token_reservation_ceiling_per_call: Literal[20000] = (
        QWEN37_FEEDBACK_INPUT_RESERVATION_TOKENS
    )
    output_token_reservation_ceiling_per_call: Literal[4106] = (
        QWEN37_FEEDBACK_OUTPUT_RESERVATION_TOKENS
    )
    output_reservation_includes_reasoning_and_answer_tokens: Literal[True] = True
    per_call_reservation_cny: Literal["0.072848000000"] = (
        QWEN37_FEEDBACK_PER_CALL_RESERVATION_CNY
    )
    maximum_reservation_cny: Literal["3.496704000000"] = (
        QWEN37_FEEDBACK_MAXIMUM_RESERVATION_CNY
    )
    phase_hard_cap_cny: Literal["4.000000000000"] = QWEN37_FEEDBACK_PHASE_HARD_CAP_CNY
    over_budget_policy: Literal["fail_closed_before_provider_call"] = (
        "fail_closed_before_provider_call"
    )
    parent_membership_processor: Literal["dashscope-qwen-assistant"] = (
        "dashscope-qwen-assistant"
    )
    parent_authority_used_for_role_permission: Literal[False] = False
    parent_remote_authorization_id: str
    parent_remote_authorization_file_sha256: Sha256
    parent_remote_receipt_file_sha256: Sha256
    parent_remote_receipt_sha256: Sha256
    parent_remote_catalog_sha256: Sha256
    model_source_lock_file_sha256: Sha256
    model_source_lock_sha256: Sha256
    pricing_lock_file_sha256: Sha256
    pricing_lock_sha256: Sha256
    role_selection_file_sha256: Sha256
    role_selection_sha256: Sha256
    cloud_upload_allowed: Literal[True] = True
    remote_model_inference_allowed: Literal[True] = True
    redistribution_allowed: Literal[False] = False
    public_demo_allowed: Literal[False] = False
    selected_assets: tuple[SelectedQwenFeedbackAssetV1, ...]
    selected_asset_set_sha256: Sha256
    authorization_sha256: Sha256

    @field_validator("selected_assets", mode="before")
    @classmethod
    def _assets_tuple(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @field_validator(
        "authorization_id",
        "reviewer_id",
        "owner_statement",
        "parent_remote_authorization_id",
    )
    @classmethod
    def _texts(cls, value: str, info) -> str:
        value = _canonical_text(value, info.field_name)
        if info.field_name == "authorization_id" and not _AUTHORIZATION_ID_RE.fullmatch(
            value
        ):
            raise ValueError("authorization_id must be canonical")
        return value

    @model_validator(mode="after")
    def _validate_authorization(self) -> Self:
        if self.reviewed_at.tzinfo is None or self.reviewed_at.utcoffset() is None:
            raise ValueError("Qwen Feedback reviewed_at needs a timezone")
        if len(self.selected_assets) != QWEN37_FEEDBACK_SELECTED_COUNT:
            raise ValueError("Qwen Feedback authorization must bind exactly 48 assets")
        if tuple(item.query_id for item in self.selected_assets) != tuple(
            sorted(item.query_id for item in self.selected_assets)
        ):
            raise ValueError("Qwen Feedback authorized assets must be query-sorted")
        if (
            len({item.query_id for item in self.selected_assets}) != 48
            or len({item.asset_id for item in self.selected_assets}) != 48
        ):
            raise ValueError("Qwen Feedback authorized identities must be unique")
        payload = [item.model_dump(mode="json") for item in self.selected_assets]
        if self.selected_asset_set_sha256 != sha256_bytes(
            canonical_json_bytes(payload)
        ):
            raise ValueError("Qwen Feedback authorized asset set hash mismatch")
        if self.authorization_sha256 != _self_hash(self, "authorization_sha256"):
            raise ValueError("Qwen Feedback authorization self hash mismatch")
        if (
            self.model_source_lock_file_sha256,
            self.model_source_lock_sha256,
            self.pricing_lock_file_sha256,
            self.pricing_lock_sha256,
        ) != (
            QWEN37_FEEDBACK_SOURCE_LOCK_FILE_SHA256,
            QWEN37_FEEDBACK_SOURCE_LOCK_SHA256,
            QWEN37_FEEDBACK_PRICING_LOCK_FILE_SHA256,
            QWEN37_FEEDBACK_PRICING_LOCK_SHA256,
        ):
            raise ValueError("Qwen Feedback source or pricing identity drifted")
        if (
            self.role_selection_file_sha256,
            self.role_selection_sha256,
        ) != (
            QWEN37_FEEDBACK_ROLE_SELECTION_FILE_SHA256,
            QWEN37_FEEDBACK_ROLE_SELECTION_SHA256,
        ):
            raise ValueError("Qwen Feedback role selection identity drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


class PortfolioS1QwenFeedbackLaunchLockV1(_StrictFrozenModel):
    """Pre-provider lock for one exact selected48 Qwen Feedback run."""

    schema_version: Literal[1] = 1
    kind: Literal["portfolio-s1-qwen37-feedback-launch-lock"] = (
        "portfolio-s1-qwen37-feedback-launch-lock"
    )
    policy_version: Literal["portfolio-s1-qwen37-feedback-launch-lock-v1"] = (
        QWEN37_FEEDBACK_LAUNCH_POLICY_VERSION
    )
    run_id: str
    status: Literal["prepared-no-provider-calls"] = "prepared-no-provider-calls"
    provider: Literal["qwen"] = "qwen"
    model: Literal["qwen3.7-plus-2026-05-26"] = QWEN37_FEEDBACK_MODEL
    processor: Literal["dashscope-qwen37-feedback"] = QWEN37_FEEDBACK_PROCESSOR
    cache_namespace: Literal["feedback-evaluator-v9"] = QWEN37_FEEDBACK_CACHE_NAMESPACE
    endpoint_configuration: Literal["DASHSCOPE_BASE_URL"] = (
        QWEN37_FEEDBACK_ENDPOINT_CONFIGURATION
    )
    requested_response_format: Literal["json_schema"] = "json_schema"
    requested_json_schema_policy_version: Literal[
        "visual-feedback-output-json-schema-v1"
    ] = QWEN37_FEEDBACK_JSON_SCHEMA_POLICY_VERSION
    requested_json_schema_name: Literal["visual_feedback_output_v1"] = (
        QWEN37_FEEDBACK_JSON_SCHEMA_NAME
    )
    requested_json_schema_sha256: Literal[
        "af673bc72a52788b4a3871b030e95123e070d23cc51388e4abb06d0d5fa9667e"
    ] = QWEN37_FEEDBACK_JSON_SCHEMA_SHA256
    requested_json_schema_strict: Literal[True] = True
    transport_policy_version: Literal[
        "visual-feedback-qwen-dashscope-json-schema-v5"
    ] = QWEN37_FEEDBACK_TRANSPORT_POLICY_VERSION
    transport_policy_sha256: Literal[
        "092f36be5eb08e4fd58edc888fe273bee2b4295059048dbd9715d472e534f20c"
    ] = QWEN37_FEEDBACK_TRANSPORT_POLICY_SHA256
    requested_stream: Literal[False] = False
    requested_thinking: Literal[True] = True
    requested_thinking_budget: Literal[2048] = QWEN37_FEEDBACK_THINKING_BUDGET
    requested_timeout_seconds: Literal[600] = QWEN37_FEEDBACK_TIMEOUT_SECONDS
    requested_temperature: None = None
    requested_top_p: None = None
    requested_seed: None = None
    requested_max_tokens: None = None
    max_completion_tokens: Literal[4096] = QWEN37_FEEDBACK_MAX_COMPLETION_TOKENS
    max_completion_tokens_documented_upper_tolerance_tokens: Literal[10] = (
        QWEN37_FEEDBACK_MAX_COMPLETION_TOKENS_TOLERANCE
    )
    provider_call_ceiling: Literal[48] = QWEN37_FEEDBACK_PROVIDER_CALL_CEILING
    max_attempts_per_selected_query: Literal[1] = 1
    retry_policy: Literal["no_retry"] = "no_retry"
    feedback_concurrency: Literal[2] = 2
    canary_call_count: Literal[6] = 6
    selection_sha256: Sha256
    corpus_sha256: Sha256
    authorization_file_sha256: Sha256
    authorization_sha256: Sha256
    model_source_lock_file_sha256: Sha256
    model_source_lock_sha256: Sha256
    pricing_lock_file_sha256: Sha256
    pricing_lock_sha256: Sha256
    role_selection_file_sha256: Sha256
    role_selection_sha256: Sha256
    control_file_sha256: Sha256
    control_sha256: Sha256
    pricing_tier_max_input_tokens: Literal[256000] = (
        QWEN37_FEEDBACK_PRICING_TIER_MAX_INPUT_TOKENS
    )
    input_token_reservation_ceiling_per_call: Literal[20000] = (
        QWEN37_FEEDBACK_INPUT_RESERVATION_TOKENS
    )
    output_token_reservation_ceiling_per_call: Literal[4106] = (
        QWEN37_FEEDBACK_OUTPUT_RESERVATION_TOKENS
    )
    output_reservation_includes_reasoning_and_answer_tokens: Literal[True] = True
    max_completion_tokens_owner_locked_despite_structured_output_guidance: Literal[
        True
    ] = True
    input_cny_per_million_tokens: Literal[2] = (
        QWEN37_FEEDBACK_INPUT_CNY_PER_MILLION_TOKENS
    )
    output_cny_per_million_tokens: Literal[8] = (
        QWEN37_FEEDBACK_OUTPUT_CNY_PER_MILLION_TOKENS
    )
    per_call_reservation_cny: Literal["0.072848000000"] = (
        QWEN37_FEEDBACK_PER_CALL_RESERVATION_CNY
    )
    maximum_reservation_cny: Literal["3.496704000000"] = (
        QWEN37_FEEDBACK_MAXIMUM_RESERVATION_CNY
    )
    phase_hard_cap_cny: Literal["4.000000000000"] = QWEN37_FEEDBACK_PHASE_HARD_CAP_CNY
    over_budget_policy: Literal["fail_closed_before_provider_call"] = (
        "fail_closed_before_provider_call"
    )
    above_pricing_tier_policy: Literal["fail_closed_before_provider_call"] = (
        "fail_closed_before_provider_call"
    )
    provider_calls_performed: Literal[0] = 0
    launch_lock_sha256: Sha256

    @field_validator("run_id")
    @classmethod
    def _run_id(cls, value: str) -> str:
        value = _canonical_text(value, "run_id")
        if not _AUTHORIZATION_ID_RE.fullmatch(value):
            raise ValueError("run_id must be canonical")
        return value

    @model_validator(mode="after")
    def _validate_launch(self) -> Self:
        if self.authorization_file_sha256 == self.authorization_sha256:
            # File/content hashes may coincide for non-self-hashed formats, but this
            # authorization always embeds a self hash and therefore must differ.
            raise ValueError("Qwen authorization file and self hashes must differ")
        if self.launch_lock_sha256 != _self_hash(self, "launch_lock_sha256"):
            raise ValueError("Qwen Feedback launch lock self hash mismatch")
        if (
            self.model_source_lock_file_sha256,
            self.model_source_lock_sha256,
            self.pricing_lock_file_sha256,
            self.pricing_lock_sha256,
        ) != (
            QWEN37_FEEDBACK_SOURCE_LOCK_FILE_SHA256,
            QWEN37_FEEDBACK_SOURCE_LOCK_SHA256,
            QWEN37_FEEDBACK_PRICING_LOCK_FILE_SHA256,
            QWEN37_FEEDBACK_PRICING_LOCK_SHA256,
        ):
            raise ValueError("Qwen Feedback launch source or pricing drifted")
        if (
            self.role_selection_file_sha256,
            self.role_selection_sha256,
        ) != (
            QWEN37_FEEDBACK_ROLE_SELECTION_FILE_SHA256,
            QWEN37_FEEDBACK_ROLE_SELECTION_SHA256,
        ):
            raise ValueError("Qwen Feedback launch role selection drifted")
        return self

    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.model_dump(mode="json"))


def _selection_assets(selection: _Selection) -> tuple[SelectedQwenFeedbackAssetV1, ...]:
    selection_sha256 = getattr(selection, "selection_sha256", None)
    entries = getattr(selection, "entries", None)
    if not isinstance(selection_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", selection_sha256
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen Feedback selection lacks a canonical self hash"
        )
    if not isinstance(entries, tuple) or len(entries) != 48:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen Feedback selection must contain exactly 48 entries"
        )
    try:
        assets = tuple(
            SelectedQwenFeedbackAssetV1(
                query_id=entry.query_id,
                asset_id=entry.asset_id,
                image_sha256=entry.image_sha256,
            )
            for entry in sorted(entries, key=lambda item: item.query_id)
        )
    except (AttributeError, ValueError) as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen Feedback selection entry binding is invalid"
        ) from error
    return assets


def validate_selected_qwen_feedback_authorization(
    authorization: PortfolioS1QwenFeedbackAuthorizationV3,
    selection: _Selection,
) -> None:
    assets = _selection_assets(selection)
    expected_set_hash = sha256_bytes(
        canonical_json_bytes([item.model_dump(mode="json") for item in assets])
    )
    if (
        type(authorization) is not PortfolioS1QwenFeedbackAuthorizationV3
        or authorization.selection_sha256 != selection.selection_sha256
        or authorization.selected_assets != assets
        or authorization.selected_asset_set_sha256 != expected_set_hash
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen Feedback authorization differs from the frozen selected48"
        )


def _require_frozen_source_and_pricing_locks(
    *,
    model_source_lock: Qwen37FeedbackModelSourceLockV1,
    model_source_lock_file_sha256: str,
    pricing_lock: Qwen37FeedbackPricingLockV1,
    pricing_lock_file_sha256: str,
) -> None:
    if (
        type(model_source_lock) is not Qwen37FeedbackModelSourceLockV1
        or model_source_lock_file_sha256 != QWEN37_FEEDBACK_SOURCE_LOCK_FILE_SHA256
        or model_source_lock.source_lock_sha256 != QWEN37_FEEDBACK_SOURCE_LOCK_SHA256
        or sha256_bytes(model_source_lock.canonical_bytes())
        != model_source_lock_file_sha256
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen model source lock differs from the frozen active identity"
        )
    if (
        type(pricing_lock) is not Qwen37FeedbackPricingLockV1
        or pricing_lock_file_sha256 != QWEN37_FEEDBACK_PRICING_LOCK_FILE_SHA256
        or pricing_lock.pricing_lock_sha256 != QWEN37_FEEDBACK_PRICING_LOCK_SHA256
        or sha256_bytes(pricing_lock.canonical_bytes()) != pricing_lock_file_sha256
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen pricing lock differs from the frozen active identity"
        )


def _require_frozen_role_selection(
    *, role_selection_file_sha256: str, role_selection_sha256: str
) -> None:
    if (
        role_selection_file_sha256,
        role_selection_sha256,
    ) != (
        QWEN37_FEEDBACK_ROLE_SELECTION_FILE_SHA256,
        QWEN37_FEEDBACK_ROLE_SELECTION_SHA256,
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen Feedback role selection differs from frozen v8"
        )


def build_selected_qwen_feedback_authorization(
    selection: _Selection,
    parent_remote_runtime: VerifiedPortfolioRemoteProcessingRuntime,
    *,
    authorization_id: str,
    reviewer_id: str,
    reviewed_at: datetime,
    owner_statement: str,
    model_source_lock: Qwen37FeedbackModelSourceLockV1,
    model_source_lock_file_sha256: str,
    pricing_lock: Qwen37FeedbackPricingLockV1,
    pricing_lock_file_sha256: str,
    role_selection_file_sha256: str,
    role_selection_sha256: str,
) -> PortfolioS1QwenFeedbackAuthorizationV3:
    """Build exact selected48 authority; never infer Qwen Feedback from Assistant."""

    parent = require_verified_portfolio_remote_processing_runtime(
        parent_remote_runtime,
        processor="dashscope-qwen-assistant",
        catalog_sha256=parent_remote_runtime.catalog.catalog_sha256,
    )
    assets = _selection_assets(selection)
    parent.catalog.require_verified_files()
    catalog_by_id = {item.asset_id: item for item in parent.catalog.assets}
    for item in assets:
        catalog_asset = catalog_by_id.get(item.asset_id)
        if (
            catalog_asset is None
            or catalog_asset.sha256 != item.image_sha256
            or catalog_asset.cloud_upload_allowed is not True
        ):
            raise PortfolioS1QwenFeedbackGovernanceError(
                "selected48 Qwen asset differs from verified Core catalog membership"
            )
    _require_frozen_source_and_pricing_locks(
        model_source_lock=model_source_lock,
        model_source_lock_file_sha256=model_source_lock_file_sha256,
        pricing_lock=pricing_lock,
        pricing_lock_file_sha256=pricing_lock_file_sha256,
    )
    _require_frozen_role_selection(
        role_selection_file_sha256=role_selection_file_sha256,
        role_selection_sha256=role_selection_sha256,
    )
    selected_payload = [item.model_dump(mode="json") for item in assets]
    draft = PortfolioS1QwenFeedbackAuthorizationV3.model_construct(
        authorization_id=authorization_id,
        reviewer_id=reviewer_id,
        reviewed_at=reviewed_at,
        owner_statement=owner_statement,
        selection_sha256=selection.selection_sha256,
        parent_remote_authorization_id=parent.authorization.authorization_id,
        parent_remote_authorization_file_sha256=parent.authorization_file_sha256,
        parent_remote_receipt_file_sha256=parent.receipt_file_sha256,
        parent_remote_receipt_sha256=parent.receipt.receipt_sha256,
        parent_remote_catalog_sha256=parent.catalog.catalog_sha256,
        model_source_lock_file_sha256=model_source_lock_file_sha256,
        model_source_lock_sha256=model_source_lock.source_lock_sha256,
        pricing_lock_file_sha256=pricing_lock_file_sha256,
        pricing_lock_sha256=pricing_lock.pricing_lock_sha256,
        role_selection_file_sha256=role_selection_file_sha256,
        role_selection_sha256=role_selection_sha256,
        selected_assets=assets,
        selected_asset_set_sha256=sha256_bytes(canonical_json_bytes(selected_payload)),
        authorization_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"authorization_sha256"})
    authorization = PortfolioS1QwenFeedbackAuthorizationV3.model_validate_json(
        canonical_json_bytes(
            {
                **unsigned,
                "authorization_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
            }
        ),
        strict=True,
    )
    validate_selected_qwen_feedback_authorization(authorization, selection)
    return authorization


def load_selected_qwen_feedback_authorization(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> PortfolioS1QwenFeedbackAuthorizationV3:
    content = read_stable_regular_file(
        path,
        label="selected Qwen3.7 Feedback authorization",
        max_bytes=2 * 1024 * 1024,
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "selected Qwen3.7 Feedback authorization file SHA-256 mismatch"
        )
    try:
        authorization = PortfolioS1QwenFeedbackAuthorizationV3.model_validate_json(
            content, strict=True
        )
    except ValueError as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "selected Qwen3.7 Feedback authorization is invalid"
        ) from error
    if authorization.canonical_bytes() != content:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "selected Qwen3.7 Feedback authorization is not canonical JSON"
        )
    return authorization


def build_qwen37_feedback_launch_lock(
    *,
    run_id: str,
    selection_sha256: str,
    corpus_sha256: str,
    authorization: PortfolioS1QwenFeedbackAuthorizationV3,
    model_source_lock: Qwen37FeedbackModelSourceLockV1,
    model_source_lock_file_sha256: str,
    pricing_lock: Qwen37FeedbackPricingLockV1,
    pricing_lock_file_sha256: str,
    role_selection_file_sha256: str,
    role_selection_sha256: str,
    control_file_sha256: str,
    control_sha256: str,
) -> PortfolioS1QwenFeedbackLaunchLockV1:
    """Build a zero-call launch lock only when every governance binding agrees."""

    if (
        authorization.selection_sha256 != selection_sha256
        or authorization.model_source_lock_file_sha256 != model_source_lock_file_sha256
        or authorization.model_source_lock_sha256
        != model_source_lock.source_lock_sha256
        or authorization.pricing_lock_file_sha256 != pricing_lock_file_sha256
        or authorization.pricing_lock_sha256 != pricing_lock.pricing_lock_sha256
        or authorization.role_selection_file_sha256 != role_selection_file_sha256
        or authorization.role_selection_sha256 != role_selection_sha256
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen Feedback launch governance differs from its authorization"
        )
    _require_frozen_source_and_pricing_locks(
        model_source_lock=model_source_lock,
        model_source_lock_file_sha256=model_source_lock_file_sha256,
        pricing_lock=pricing_lock,
        pricing_lock_file_sha256=pricing_lock_file_sha256,
    )
    _require_frozen_role_selection(
        role_selection_file_sha256=role_selection_file_sha256,
        role_selection_sha256=role_selection_sha256,
    )
    draft = PortfolioS1QwenFeedbackLaunchLockV1.model_construct(
        run_id=run_id,
        selection_sha256=selection_sha256,
        corpus_sha256=corpus_sha256,
        authorization_file_sha256=sha256_bytes(authorization.canonical_bytes()),
        authorization_sha256=authorization.authorization_sha256,
        model_source_lock_file_sha256=model_source_lock_file_sha256,
        model_source_lock_sha256=model_source_lock.source_lock_sha256,
        pricing_lock_file_sha256=pricing_lock_file_sha256,
        pricing_lock_sha256=pricing_lock.pricing_lock_sha256,
        role_selection_file_sha256=role_selection_file_sha256,
        role_selection_sha256=role_selection_sha256,
        control_file_sha256=control_file_sha256,
        control_sha256=control_sha256,
        launch_lock_sha256="0" * 64,
    )
    unsigned = draft.model_dump(mode="json", exclude={"launch_lock_sha256"})
    return PortfolioS1QwenFeedbackLaunchLockV1.model_validate_json(
        canonical_json_bytes(
            {
                **unsigned,
                "launch_lock_sha256": sha256_bytes(canonical_json_bytes(unsigned)),
            }
        ),
        strict=True,
    )


def require_qwen37_feedback_pre_call_budget(
    *,
    estimated_input_tokens_including_images: int,
    provider_calls_already_reserved: int,
    committed_cost_cny: str,
) -> str:
    """Reserve one Qwen call or fail closed before any provider invocation.

    ``committed_cost_cny`` is settled actual cost plus forfeited and unresolved
    reservations.  The caller persists the returned fixed-scale reservation
    before invoking the provider.
    """

    if (
        type(estimated_input_tokens_including_images) is not int
        or estimated_input_tokens_including_images <= 0
        or estimated_input_tokens_including_images
        > QWEN37_FEEDBACK_INPUT_RESERVATION_TOKENS
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen Feedback input estimate exceeds the 20,000-token reservation"
        )
    if (
        type(provider_calls_already_reserved) is not int
        or provider_calls_already_reserved < 0
        or provider_calls_already_reserved >= QWEN37_FEEDBACK_PROVIDER_CALL_CEILING
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen Feedback provider-call ceiling is exhausted"
        )
    try:
        committed = Decimal(committed_cost_cny)
    except (InvalidOperation, TypeError) as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen Feedback committed cost is invalid"
        ) from error
    if not committed.is_finite() or committed < 0:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen Feedback committed cost is invalid"
        )
    reservation = Decimal(QWEN37_FEEDBACK_PER_CALL_RESERVATION_CNY)
    if committed + reservation > Decimal(QWEN37_FEEDBACK_PHASE_HARD_CAP_CNY):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen Feedback phase hard cap would be exceeded"
        )
    return QWEN37_FEEDBACK_PER_CALL_RESERVATION_CNY


def require_qwen37_feedback_pre_call_budget_v2(
    *,
    estimated_input_tokens_including_images: int,
    provider_calls_already_reserved: int,
    committed_cost_cny: str,
) -> str:
    """Reserve one exact240 Feedback call under the owner-approved CNY18 cap."""

    if (
        type(estimated_input_tokens_including_images) is not int
        or estimated_input_tokens_including_images <= 0
        or estimated_input_tokens_including_images
        > QWEN37_FEEDBACK_INPUT_RESERVATION_TOKENS
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen Feedback input estimate exceeds the 20,000-token reservation"
        )
    if (
        type(provider_calls_already_reserved) is not int
        or provider_calls_already_reserved < 0
        or provider_calls_already_reserved >= QWEN37_FEEDBACK_PROVIDER_CALL_CEILING_V2
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen Feedback exact240 provider-call ceiling is exhausted"
        )
    try:
        committed = Decimal(committed_cost_cny)
    except (InvalidOperation, TypeError) as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen Feedback committed cost is invalid"
        ) from error
    if not committed.is_finite() or committed < 0:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen Feedback committed cost is invalid"
        )
    reservation = Decimal(QWEN37_FEEDBACK_PER_CALL_RESERVATION_CNY)
    if committed + reservation > Decimal(QWEN37_FEEDBACK_PHASE_HARD_CAP_CNY_V2):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen Feedback exact240 phase hard cap would be exceeded"
        )
    return QWEN37_FEEDBACK_PER_CALL_RESERVATION_CNY


def require_qwen38_feedback_pre_call_budget(
    *,
    estimated_input_tokens_including_images: int,
    provider_calls_already_reserved: int,
    committed_cost_cny: str,
) -> str:
    """Check the Qwen3.8 technical envelope after separate owner approval.

    This helper deliberately does not mint or infer owner authorization.  A
    caller must verify a separately created authorization before invoking it.
    """

    if (
        type(estimated_input_tokens_including_images) is not int
        or estimated_input_tokens_including_images <= 0
        or estimated_input_tokens_including_images
        > QWEN38_FEEDBACK_INPUT_RESERVATION_TOKENS
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8 Feedback input estimate exceeds the 20,000-token reservation"
        )
    if (
        type(provider_calls_already_reserved) is not int
        or provider_calls_already_reserved < 0
        or provider_calls_already_reserved >= QWEN38_FEEDBACK_PROVIDER_CALL_CEILING
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8 Feedback exact240 provider-call ceiling is exhausted"
        )
    try:
        committed = Decimal(committed_cost_cny)
    except (InvalidOperation, TypeError) as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8 Feedback committed cost is invalid"
        ) from error
    if not committed.is_finite() or committed < 0:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8 Feedback committed cost is invalid"
        )
    reservation = Decimal(QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY)
    if committed + reservation > Decimal(QWEN38_FEEDBACK_TECHNICAL_PHASE_HARD_CAP_CNY):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8 Feedback technical phase hard cap would be exceeded"
        )
    return QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY


def require_qwen38_feedback_pre_call_budget_v2(
    *,
    estimated_input_tokens_including_images: int,
    provider_calls_already_reserved: int,
    committed_cost_cny: str,
) -> str:
    """Check the 241-call envelope after an eligible global retry claim.

    This function checks only tokens, call ceiling, and CNY94.  Call 241 must
    additionally be backed by the outer orchestration policy's unique,
    create-only ``invalid_feedback_json`` retry claim for the same entry.
    """

    if (
        type(estimated_input_tokens_including_images) is not int
        or estimated_input_tokens_including_images <= 0
        or estimated_input_tokens_including_images
        > QWEN38_FEEDBACK_INPUT_RESERVATION_TOKENS
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8 Feedback input estimate exceeds the 20,000-token reservation"
        )
    if (
        type(provider_calls_already_reserved) is not int
        or provider_calls_already_reserved < 0
        or provider_calls_already_reserved >= QWEN38_FEEDBACK_PROVIDER_CALL_CEILING_V2
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8 Feedback 241-call provider ceiling is exhausted"
        )
    try:
        committed = Decimal(committed_cost_cny)
    except (InvalidOperation, TypeError) as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8 Feedback committed cost is invalid"
        ) from error
    if not committed.is_finite() or committed < 0:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8 Feedback committed cost is invalid"
        )
    reservation = Decimal(QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY)
    if committed + reservation > Decimal(QWEN38_FEEDBACK_TECHNICAL_PHASE_HARD_CAP_CNY):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8 Feedback technical phase hard cap would be exceeded"
        )
    return QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY


def require_qwen38_feedback_pre_call_budget_v3(
    *,
    estimated_input_tokens_including_images: int,
    provider_calls_already_reserved: int,
    committed_cost_cny: str,
) -> str:
    """Check the live 243-call fresh-v3 envelope's CNY113 technical stop.

    The active role separately records CNY150 owner authority.  This helper
    deliberately enforces the smaller technical ceiling and never widens it.
    """

    if (
        type(estimated_input_tokens_including_images) is not int
        or estimated_input_tokens_including_images <= 0
        or estimated_input_tokens_including_images
        > QWEN38_FEEDBACK_INPUT_RESERVATION_TOKENS
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8 Feedback input estimate exceeds the 20,000-token reservation"
        )
    if (
        type(provider_calls_already_reserved) is not int
        or provider_calls_already_reserved < 0
        or provider_calls_already_reserved >= QWEN38_FEEDBACK_PROVIDER_CALL_CEILING_V3
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8 Feedback 243-call provider ceiling is exhausted"
        )
    try:
        committed = Decimal(committed_cost_cny)
    except (InvalidOperation, TypeError) as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8 Feedback committed cost is invalid"
        ) from error
    if not committed.is_finite() or committed < 0:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8 Feedback committed cost is invalid"
        )
    reservation = Decimal(QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY_V2)
    if committed + reservation > Decimal(
        QWEN38_FEEDBACK_TECHNICAL_PHASE_HARD_CAP_CNY_V2
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8 Feedback CNY113 technical phase hard cap would be exceeded"
        )
    return QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY_V2


def require_qwen38_feedback_pre_call_budget_v4(
    *,
    estimated_input_tokens_including_images: int,
    provider_calls_already_reserved: int,
    committed_cumulative_cost_cny: str,
) -> str:
    """Check the live 15-call schema canary under its fresh CNY10 stop."""

    if (
        type(estimated_input_tokens_including_images) is not int
        or estimated_input_tokens_including_images <= 0
        or estimated_input_tokens_including_images
        > QWEN38_FEEDBACK_INPUT_RESERVATION_TOKENS
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8 Feedback input estimate exceeds the 20,000-token reservation"
        )
    if (
        type(provider_calls_already_reserved) is not int
        or provider_calls_already_reserved < 0
        or provider_calls_already_reserved
        >= QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_PROVIDER_CALL_CEILING
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8 Feedback schema-canary 15-call provider ceiling is exhausted"
        )
    try:
        committed = Decimal(committed_cumulative_cost_cny)
    except (InvalidOperation, TypeError) as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8 Feedback cumulative committed cost is invalid"
        ) from error
    if not committed.is_finite() or committed < Decimal(
        QWEN38_FEEDBACK_ROUND3_SCHEMA_PRIOR_ACTUAL_COST_CNY
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8 Feedback cumulative committed cost is invalid"
        )
    reservation = Decimal(QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY_V2)
    if committed + reservation > Decimal(
        QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_CUMULATIVE_HARD_CAP_CNY
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8 Feedback schema-canary CNY10 fresh stage hard cap would be exceeded"
        )
    return QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY_V2


def require_qwen38_feedback_pre_call_budget_v5(
    *,
    estimated_input_tokens_including_images: int,
    provider_calls_already_reserved: int,
    committed_cumulative_cost_cny: str,
) -> str:
    """Reserve one call only while the approved phase60 envelope remains live."""

    if (
        type(estimated_input_tokens_including_images) is not int
        or estimated_input_tokens_including_images <= 0
        or estimated_input_tokens_including_images
        > QWEN38_FEEDBACK_INPUT_RESERVATION_TOKENS
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8 Feedback input estimate exceeds the 20,000-token reservation"
        )
    if (
        type(provider_calls_already_reserved) is not int
        or provider_calls_already_reserved < 0
        or provider_calls_already_reserved
        >= QWEN38_FEEDBACK_PHASE60_NEW_PROVIDER_CALL_CEILING
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8 Feedback phase60 60-call provider ceiling is exhausted"
        )
    try:
        committed = Decimal(committed_cumulative_cost_cny)
    except (InvalidOperation, TypeError) as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8 Feedback phase60 cumulative committed cost is invalid"
        ) from error
    if not committed.is_finite() or committed < Decimal(
        QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_CUMULATIVE_ACTUAL_CNY
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8 Feedback phase60 cumulative committed cost is invalid"
        )
    reservation = Decimal(QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY_V2)
    fresh_reserved_after_call = reservation * (provider_calls_already_reserved + 1)
    if fresh_reserved_after_call > Decimal(
        QWEN38_FEEDBACK_PHASE60_FRESH_MAXIMUM_RESERVATION_CNY
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8 Feedback phase60 CNY27.692640 maximum reservation would be "
            "exceeded"
        )
    if committed + reservation > Decimal(
        QWEN38_FEEDBACK_PHASE60_CUMULATIVE_TECHNICAL_HARD_CAP_CNY
    ):
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8 Feedback phase60 CNY28 fresh technical cap would be exceeded"
        )
    return QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY_V2


def load_qwen37_feedback_launch_lock(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> PortfolioS1QwenFeedbackLaunchLockV1:
    content = read_stable_regular_file(
        path, label="Qwen3.7 Feedback launch lock", max_bytes=1024 * 1024
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.7 Feedback launch lock file SHA-256 mismatch"
        )
    try:
        lock = PortfolioS1QwenFeedbackLaunchLockV1.model_validate_json(
            content, strict=True
        )
    except ValueError as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.7 Feedback launch lock is invalid"
        ) from error
    if lock.canonical_bytes() != content:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.7 Feedback launch lock is not canonical JSON"
        )
    return lock


def load_qwen37_feedback_model_source_lock(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> Qwen37FeedbackModelSourceLockV1:
    content = read_stable_regular_file(
        path, label="Qwen3.7 Feedback model source lock", max_bytes=1024 * 1024
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.7 Feedback model source lock file SHA-256 mismatch"
        )
    try:
        lock = Qwen37FeedbackModelSourceLockV1.model_validate_json(content, strict=True)
    except ValueError as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.7 Feedback model source lock is invalid"
        ) from error
    if lock.canonical_bytes() != content:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.7 Feedback model source lock is not canonical JSON"
        )
    return lock


def load_qwen37_feedback_pricing_lock(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> Qwen37FeedbackPricingLockV1:
    content = read_stable_regular_file(
        path, label="Qwen3.7 Feedback pricing lock", max_bytes=1024 * 1024
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.7 Feedback pricing lock file SHA-256 mismatch"
        )
    try:
        lock = Qwen37FeedbackPricingLockV1.model_validate_json(content, strict=True)
    except ValueError as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.7 Feedback pricing lock is invalid"
        ) from error
    if lock.canonical_bytes() != content:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.7 Feedback pricing lock is not canonical JSON"
        )
    return lock


def load_qwen37_feedback_pricing_lock_v2(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> Qwen37FeedbackPricingLockV2:
    content = read_stable_regular_file(
        path, label="Qwen3.7 Feedback exact240 pricing lock", max_bytes=1024 * 1024
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.7 Feedback exact240 pricing lock file SHA-256 mismatch"
        )
    try:
        lock = Qwen37FeedbackPricingLockV2.model_validate_json(content, strict=True)
    except ValueError as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.7 Feedback exact240 pricing lock is invalid"
        ) from error
    if lock.canonical_bytes() != content:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.7 Feedback exact240 pricing lock is not canonical JSON"
        )
    return lock


def load_qwen38_feedback_model_source_lock(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> Qwen38FeedbackModelSourceLockV1:
    content = read_stable_regular_file(
        path, label="Qwen3.8-Max Feedback model source lock", max_bytes=1024 * 1024
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback model source lock file SHA-256 mismatch"
        )
    try:
        lock = Qwen38FeedbackModelSourceLockV1.model_validate_json(content, strict=True)
    except ValueError as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback model source lock is invalid"
        ) from error
    if lock.canonical_bytes() != content:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback model source lock is not canonical JSON"
        )
    return lock


def load_qwen38_feedback_model_source_lock_v2(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> Qwen38FeedbackModelSourceLockV2:
    content = read_stable_regular_file(
        path,
        label="Qwen3.8-Max Feedback 6144-token model source lock",
        max_bytes=1024 * 1024,
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback model source lock v2 file SHA-256 mismatch"
        )
    try:
        lock = Qwen38FeedbackModelSourceLockV2.model_validate_json(content, strict=True)
    except ValueError as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback model source lock v2 is invalid"
        ) from error
    if lock.canonical_bytes() != content:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback model source lock v2 is not canonical JSON"
        )
    return lock


def load_qwen38_feedback_model_source_lock_v3(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> Qwen38FeedbackModelSourceLockV3:
    content = read_stable_regular_file(
        path,
        label="Qwen3.8-Max Feedback Round3 JSON-object model source lock",
        max_bytes=1024 * 1024,
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback model source lock v3 file SHA-256 mismatch"
        )
    try:
        lock = Qwen38FeedbackModelSourceLockV3.model_validate_json(content, strict=True)
    except ValueError as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback model source lock v3 is invalid"
        ) from error
    if lock.canonical_bytes() != content:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback model source lock v3 is not canonical JSON"
        )
    return lock


def load_qwen38_feedback_model_source_lock_v4(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> Qwen38FeedbackModelSourceLockV4:
    content = read_stable_regular_file(
        path,
        label="Qwen3.8-Max Feedback model source lock v4",
        max_bytes=1024 * 1024,
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback model source lock v4 file SHA-256 mismatch"
        )
    try:
        lock = Qwen38FeedbackModelSourceLockV4.model_validate_json(content, strict=True)
    except ValueError as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback model source lock v4 is invalid"
        ) from error
    if lock.canonical_bytes() != content:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback model source lock v4 is not canonical JSON"
        )
    return lock


def load_qwen38_feedback_model_source_lock_v5(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> Qwen38FeedbackModelSourceLockV5:
    content = read_stable_regular_file(
        path,
        label="Qwen3.8-Max Feedback phase60 live model source lock v5",
        max_bytes=1024 * 1024,
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback model source lock v5 file SHA-256 mismatch"
        )
    try:
        lock = Qwen38FeedbackModelSourceLockV5.model_validate_json(content, strict=True)
    except ValueError as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback model source lock v5 is invalid"
        ) from error
    if lock.canonical_bytes() != content:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback model source lock v5 is not canonical JSON"
        )
    return lock


def load_qwen38_feedback_pricing_lock_v3(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> Qwen38FeedbackPricingLockV3:
    content = read_stable_regular_file(
        path, label="Qwen3.8-Max Feedback exact240 pricing lock", max_bytes=1024 * 1024
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback exact240 pricing lock file SHA-256 mismatch"
        )
    try:
        lock = Qwen38FeedbackPricingLockV3.model_validate_json(content, strict=True)
    except ValueError as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback exact240 pricing lock is invalid"
        ) from error
    if lock.canonical_bytes() != content:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback exact240 pricing lock is not canonical JSON"
        )
    return lock


def load_qwen38_feedback_pricing_lock_v4(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> Qwen38FeedbackPricingLockV4:
    content = read_stable_regular_file(
        path,
        label="Qwen3.8-Max Feedback global-retry pricing lock",
        max_bytes=1024 * 1024,
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback global-retry pricing lock file SHA-256 mismatch"
        )
    try:
        lock = Qwen38FeedbackPricingLockV4.model_validate_json(content, strict=True)
    except ValueError as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback global-retry pricing lock is invalid"
        ) from error
    if lock.canonical_bytes() != content:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback global-retry pricing lock is not canonical JSON"
        )
    return lock


def load_qwen38_feedback_pricing_lock_v5(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> Qwen38FeedbackPricingLockV5:
    content = read_stable_regular_file(
        path,
        label="Qwen3.8-Max Feedback fresh-v3 pricing lock",
        max_bytes=1024 * 1024,
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback fresh-v3 pricing lock file SHA-256 mismatch"
        )
    try:
        lock = Qwen38FeedbackPricingLockV5.model_validate_json(content, strict=True)
    except ValueError as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback fresh-v3 pricing lock is invalid"
        ) from error
    if lock.canonical_bytes() != content:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback fresh-v3 pricing lock is not canonical JSON"
        )
    return lock


def load_qwen38_feedback_pricing_lock_v6(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> Qwen38FeedbackPricingLockV6:
    content = read_stable_regular_file(
        path,
        label="Qwen3.8-Max Feedback Round3 cumulative pricing lock",
        max_bytes=1024 * 1024,
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback pricing lock v6 file SHA-256 mismatch"
        )
    try:
        lock = Qwen38FeedbackPricingLockV6.model_validate_json(content, strict=True)
    except ValueError as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback pricing lock v6 is invalid"
        ) from error
    if lock.canonical_bytes() != content:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback pricing lock v6 is not canonical JSON"
        )
    return lock


def load_qwen38_feedback_pricing_lock_v7(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> Qwen38FeedbackPricingLockV7:
    content = read_stable_regular_file(
        path,
        label="Qwen3.8-Max Feedback Round3 schema pricing lock",
        max_bytes=1024 * 1024,
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback pricing lock v7 file SHA-256 mismatch"
        )
    try:
        lock = Qwen38FeedbackPricingLockV7.model_validate_json(content, strict=True)
    except ValueError as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback pricing lock v7 is invalid"
        ) from error
    if lock.canonical_bytes() != content:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback pricing lock v7 is not canonical JSON"
        )
    return lock


def load_qwen38_feedback_pricing_lock_v8(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> Qwen38FeedbackPricingLockV8:
    content = read_stable_regular_file(
        path,
        label="Qwen3.8-Max Feedback phase60 live pricing lock v8",
        max_bytes=1024 * 1024,
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback pricing lock v8 file SHA-256 mismatch"
        )
    try:
        lock = Qwen38FeedbackPricingLockV8.model_validate_json(content, strict=True)
    except ValueError as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback pricing lock v8 is invalid"
        ) from error
    if lock.canonical_bytes() != content:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback pricing lock v8 is not canonical JSON"
        )
    return lock


def load_qwen38_feedback_role_selection_v12(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> Qwen38FeedbackRoleSelectionV12:
    content = read_stable_regular_file(
        path,
        label="Qwen3.8-Max Feedback role selection v12",
        max_bytes=1024 * 1024,
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback role selection v12 file SHA-256 mismatch"
        )
    try:
        lock = Qwen38FeedbackRoleSelectionV12.model_validate_json(content, strict=True)
    except ValueError as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback role selection v12 is invalid"
        ) from error
    if lock.canonical_bytes() != content:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback role selection v12 is not canonical JSON"
        )
    return lock


def load_qwen38_feedback_role_selection_v13(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> Qwen38FeedbackRoleSelectionV13:
    content = read_stable_regular_file(
        path,
        label="Qwen3.8-Max Feedback role selection v13",
        max_bytes=1024 * 1024,
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback role selection v13 file SHA-256 mismatch"
        )
    try:
        lock = Qwen38FeedbackRoleSelectionV13.model_validate_json(content, strict=True)
    except ValueError as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback role selection v13 is invalid"
        ) from error
    if lock.canonical_bytes() != content:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback role selection v13 is not canonical JSON"
        )
    return lock


def load_qwen38_feedback_role_selection_v14(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> Qwen38FeedbackRoleSelectionV14:
    content = read_stable_regular_file(
        path,
        label="Qwen3.8-Max Feedback role selection v14",
        max_bytes=1024 * 1024,
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback role selection v14 file SHA-256 mismatch"
        )
    try:
        lock = Qwen38FeedbackRoleSelectionV14.model_validate_json(content, strict=True)
    except ValueError as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback role selection v14 is invalid"
        ) from error
    if lock.canonical_bytes() != content:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback role selection v14 is not canonical JSON"
        )
    return lock


def load_qwen38_feedback_role_selection_v15(
    path: str | Path,
    *,
    expected_file_sha256: str,
) -> Qwen38FeedbackRoleSelectionV15:
    content = read_stable_regular_file(
        path,
        label="Qwen3.8-Max Feedback role selection v15",
        max_bytes=1024 * 1024,
    )
    if sha256_bytes(content) != expected_file_sha256:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback role selection v15 file SHA-256 mismatch"
        )
    try:
        lock = Qwen38FeedbackRoleSelectionV15.model_validate_json(content, strict=True)
    except ValueError as error:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback role selection v15 is invalid"
        ) from error
    if lock.canonical_bytes() != content:
        raise PortfolioS1QwenFeedbackGovernanceError(
            "Qwen3.8-Max Feedback role selection v15 is not canonical JSON"
        )
    return lock


def write_selected_qwen_feedback_authorization(
    path: str | Path,
    authorization: PortfolioS1QwenFeedbackAuthorizationV3,
) -> Path:
    return atomic_create_file(path, authorization.canonical_bytes())


def write_qwen37_feedback_launch_lock(
    path: str | Path,
    launch_lock: PortfolioS1QwenFeedbackLaunchLockV1,
) -> Path:
    return atomic_create_file(path, launch_lock.canonical_bytes())


__all__ = [
    "PortfolioS1QwenFeedbackAuthorizationV3",
    "PortfolioS1QwenFeedbackGovernanceError",
    "PortfolioS1QwenFeedbackLaunchLockV1",
    "QWEN37_FEEDBACK_AUTHORIZATION_POLICY_VERSION",
    "QWEN37_FEEDBACK_CACHE_NAMESPACE",
    "QWEN37_FEEDBACK_CONTEXT_WINDOW_TOKENS",
    "QWEN37_FEEDBACK_ENDPOINT",
    "QWEN37_FEEDBACK_ENDPOINT_CONFIGURATION",
    "QWEN37_FEEDBACK_INPUT_CNY_PER_MILLION_TOKENS",
    "QWEN37_FEEDBACK_INPUT_RESERVATION_TOKENS",
    "QWEN37_FEEDBACK_JSON_SCHEMA_NAME",
    "QWEN37_FEEDBACK_JSON_SCHEMA_POLICY_VERSION",
    "QWEN37_FEEDBACK_JSON_SCHEMA_SHA256",
    "QWEN37_FEEDBACK_LAUNCH_POLICY_VERSION",
    "QWEN37_FEEDBACK_MAX_COMPLETION_TOKENS",
    "QWEN37_FEEDBACK_MAX_COMPLETION_TOKENS_TOLERANCE",
    "QWEN37_FEEDBACK_MODEL",
    "QWEN37_FEEDBACK_OUTPUT_CNY_PER_MILLION_TOKENS",
    "QWEN37_FEEDBACK_OUTPUT_RESERVATION_TOKENS",
    "QWEN37_FEEDBACK_PER_CALL_RESERVATION_CNY",
    "QWEN37_FEEDBACK_MAXIMUM_RESERVATION_CNY",
    "QWEN37_FEEDBACK_MAXIMUM_RESERVATION_CNY_V2",
    "QWEN37_FEEDBACK_PHASE_HARD_CAP_CNY",
    "QWEN37_FEEDBACK_PHASE_HARD_CAP_CNY_V2",
    "QWEN37_FEEDBACK_PRICING_LOCK_POLICY_VERSION",
    "QWEN37_FEEDBACK_PRICING_LOCK_POLICY_VERSION_V2",
    "QWEN37_FEEDBACK_PRICING_LOCK_FILE_SHA256",
    "QWEN37_FEEDBACK_PRICING_LOCK_FILE_SHA256_V2",
    "QWEN37_FEEDBACK_PRICING_LOCK_SHA256",
    "QWEN37_FEEDBACK_PRICING_LOCK_SHA256_V2",
    "QWEN37_FEEDBACK_PRICING_TIER_MAX_INPUT_TOKENS",
    "QWEN37_FEEDBACK_PROCESSOR",
    "QWEN37_FEEDBACK_PROVIDER",
    "QWEN37_FEEDBACK_PROVIDER_CALL_CEILING",
    "QWEN37_FEEDBACK_PROVIDER_CALL_CEILING_V2",
    "QWEN37_FEEDBACK_PROVIDER_MAX_OUTPUT_TOKENS",
    "QWEN37_FEEDBACK_SELECTED_COUNT",
    "QWEN37_FEEDBACK_SELECTED_COUNT_V2",
    "QWEN37_FEEDBACK_ROLE_SELECTION_FILE_SHA256",
    "QWEN37_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V2",
    "QWEN37_FEEDBACK_ROLE_SELECTION_SHA256",
    "QWEN37_FEEDBACK_ROLE_SELECTION_SHA256_V2",
    "QWEN37_FEEDBACK_SOURCE_LOCK_FILE_SHA256",
    "QWEN37_FEEDBACK_SOURCE_LOCK_SHA256",
    "QWEN37_FEEDBACK_SOURCE_LOCK_POLICY_VERSION",
    "QWEN37_FEEDBACK_THINKING_BUDGET",
    "QWEN37_FEEDBACK_TIMEOUT_SECONDS",
    "QWEN37_FEEDBACK_TRANSPORT_POLICY_SHA256",
    "QWEN37_FEEDBACK_TRANSPORT_POLICY_VERSION",
    "Qwen37FeedbackModelSourceLockV1",
    "Qwen37FeedbackPricingLockV1",
    "Qwen37FeedbackPricingLockV2",
    "Qwen37FeedbackSourceEvidenceV1",
    "QWEN38_FEEDBACK_CACHE_NAMESPACE",
    "QWEN38_FEEDBACK_CACHE_NAMESPACE_V2",
    "QWEN38_FEEDBACK_CACHE_NAMESPACE_V3",
    "QWEN38_FEEDBACK_CACHE_NAMESPACE_V4",
    "QWEN38_FEEDBACK_BOUND_ARTIFACT_POLICY_VERSION_V2",
    "QWEN38_FEEDBACK_BOUND_ARTIFACT_SCHEMA_VERSION_V2",
    "QWEN38_FEEDBACK_BOUND_ARTIFACT_POLICY_VERSION_V3",
    "QWEN38_FEEDBACK_BOUND_ARTIFACT_SCHEMA_VERSION_V3",
    "QWEN38_FEEDBACK_BOUND_ARTIFACT_POLICY_VERSION_V4",
    "QWEN38_FEEDBACK_BOUND_ARTIFACT_SCHEMA_VERSION_V4",
    "QWEN38_FEEDBACK_ENDPOINT",
    "QWEN38_FEEDBACK_ENDPOINT_CONFIGURATION",
    "QWEN38_FEEDBACK_INPUT_CNY_PER_MILLION_TOKENS",
    "QWEN38_FEEDBACK_INPUT_RESERVATION_TOKENS",
    "QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V1",
    "QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_SHA256_V2",
    "QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_VERSION_V1",
    "QWEN38_FEEDBACK_GLOBAL_RETRY_POLICY_VERSION_V2",
    "QWEN38_FEEDBACK_JSON_SCHEMA_NAME",
    "QWEN38_FEEDBACK_JSON_SCHEMA_POLICY_VERSION",
    "QWEN38_FEEDBACK_JSON_SCHEMA_SHA256",
    "QWEN38_FEEDBACK_MAX_COMPLETION_TOKENS",
    "QWEN38_FEEDBACK_MAX_COMPLETION_TOKENS_V2",
    "QWEN38_FEEDBACK_MAX_COMPLETION_TOKENS_TOLERANCE",
    "QWEN38_FEEDBACK_MAXIMUM_RESERVATION_CNY",
    "QWEN38_FEEDBACK_MAXIMUM_RESERVATION_CNY_V2",
    "QWEN38_FEEDBACK_MAXIMUM_RESERVATION_CNY_V3",
    "QWEN38_FEEDBACK_MODEL",
    "QWEN38_FEEDBACK_OUTPUT_CNY_PER_MILLION_TOKENS",
    "QWEN38_FEEDBACK_OUTPUT_RESERVATION_TOKENS",
    "QWEN38_FEEDBACK_OUTPUT_RESERVATION_TOKENS_V2",
    "QWEN38_FEEDBACK_OWNER_AUTHORIZED_BUDGET_CEILING_CNY",
    "QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY",
    "QWEN38_FEEDBACK_PER_CALL_RESERVATION_CNY_V2",
    "QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V3",
    "QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V4",
    "QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V5",
    "QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V6",
    "QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V7",
    "QWEN38_FEEDBACK_PRICING_LOCK_FILE_SHA256_V8",
    "QWEN38_FEEDBACK_PRICING_LOCK_RELATIVE_PATH_V6",
    "QWEN38_FEEDBACK_PRICING_LOCK_RELATIVE_PATH_V7",
    "QWEN38_FEEDBACK_PRICING_LOCK_RELATIVE_PATH_V8",
    "QWEN38_FEEDBACK_PRICING_LOCK_POLICY_VERSION_V3",
    "QWEN38_FEEDBACK_PRICING_LOCK_POLICY_VERSION_V4",
    "QWEN38_FEEDBACK_PRICING_LOCK_POLICY_VERSION_V5",
    "QWEN38_FEEDBACK_PRICING_LOCK_POLICY_VERSION_V6",
    "QWEN38_FEEDBACK_PRICING_LOCK_POLICY_VERSION_V7",
    "QWEN38_FEEDBACK_PRICING_LOCK_POLICY_VERSION_V8",
    "QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V3",
    "QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V4",
    "QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V5",
    "QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V6",
    "QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V7",
    "QWEN38_FEEDBACK_PRICING_LOCK_SHA256_V8",
    "QWEN38_FEEDBACK_PRICING_TIER_MAX_INPUT_TOKENS",
    "QWEN38_FEEDBACK_PROCESSOR",
    "QWEN38_FEEDBACK_PROVIDER",
    "QWEN38_FEEDBACK_PROVIDER_CALL_CEILING",
    "QWEN38_FEEDBACK_PROVIDER_CALL_CEILING_V2",
    "QWEN38_FEEDBACK_PROVIDER_CALL_CEILING_V3",
    "QWEN38_FEEDBACK_RESULT_SCHEMA_VERSION_V2",
    "QWEN38_FEEDBACK_RESULT_SCHEMA_VERSION_V3",
    "QWEN38_FEEDBACK_RESULT_SCHEMA_VERSION_V4",
    "QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256",
    "QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V11",
    "QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V12",
    "QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V13",
    "QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V14",
    "QWEN38_FEEDBACK_ROLE_SELECTION_FILE_SHA256_V15",
    "QWEN38_FEEDBACK_ROLE_SELECTION_RELATIVE_PATH_V13",
    "QWEN38_FEEDBACK_ROLE_SELECTION_RELATIVE_PATH_V14",
    "QWEN38_FEEDBACK_ROLE_SELECTION_RELATIVE_PATH_V15",
    "QWEN38_FEEDBACK_ROLE_SELECTION_SHA256",
    "QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V11",
    "QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V12",
    "QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V13",
    "QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V14",
    "QWEN38_FEEDBACK_ROLE_SELECTION_SHA256_V15",
    "QWEN38_FEEDBACK_SELECTED_COUNT",
    "QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256",
    "QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V2",
    "QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V3",
    "QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V4",
    "QWEN38_FEEDBACK_SOURCE_LOCK_FILE_SHA256_V5",
    "QWEN38_FEEDBACK_SOURCE_LOCK_RELATIVE_PATH_V3",
    "QWEN38_FEEDBACK_SOURCE_LOCK_RELATIVE_PATH_V4",
    "QWEN38_FEEDBACK_SOURCE_LOCK_RELATIVE_PATH_V5",
    "QWEN38_FEEDBACK_SOURCE_LOCK_POLICY_VERSION",
    "QWEN38_FEEDBACK_SOURCE_LOCK_POLICY_VERSION_V2",
    "QWEN38_FEEDBACK_SOURCE_LOCK_POLICY_VERSION_V3",
    "QWEN38_FEEDBACK_SOURCE_LOCK_POLICY_VERSION_V4",
    "QWEN38_FEEDBACK_SOURCE_LOCK_POLICY_VERSION_V5",
    "QWEN38_FEEDBACK_SOURCE_LOCK_SHA256",
    "QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V2",
    "QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V3",
    "QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V4",
    "QWEN38_FEEDBACK_SOURCE_LOCK_SHA256_V5",
    "QWEN38_FEEDBACK_TECHNICAL_PHASE_HARD_CAP_CNY",
    "QWEN38_FEEDBACK_TECHNICAL_PHASE_HARD_CAP_CNY_V2",
    "QWEN38_FEEDBACK_ROUND3_CUMULATIVE_MAXIMUM_CNY",
    "QWEN38_FEEDBACK_ROUND3_CUMULATIVE_TECHNICAL_HARD_CAP_CNY",
    "QWEN38_FEEDBACK_ROUND3_PARSER_POLICY_SHA256_V3",
    "QWEN38_FEEDBACK_ROUND3_PARSER_POLICY_VERSION_V3",
    "QWEN38_FEEDBACK_ROUND3_PRIOR_ACTUAL_COST_CNY",
    "QWEN38_FEEDBACK_ROUND3_PROMPT_POLICY_SHA256_V6",
    "QWEN38_FEEDBACK_ROUND3_PROMPT_POLICY_VERSION_V6",
    "QWEN38_FEEDBACK_ROUND3_RETRY_POLICY_SHA256_V1",
    "QWEN38_FEEDBACK_ROUND3_RETRY_POLICY_SHA256_V2",
    "QWEN38_FEEDBACK_ROUND3_RETRY_POLICY_VERSION_V1",
    "QWEN38_FEEDBACK_ROUND3_RETRY_POLICY_VERSION_V2",
    "QWEN38_FEEDBACK_ROUND3_WIRE_KIND_V1",
    "QWEN38_FEEDBACK_ROUND3_WIRE_KIND_V2",
    "QWEN38_FEEDBACK_ROUND3_SCHEMA_PRIOR_ACTUAL_COST_CNY",
    "QWEN38_FEEDBACK_ROUND3_SCHEMA_CUMULATIVE_MAXIMUM_CNY",
    "QWEN38_FEEDBACK_ROUND3_SCHEMA_CUMULATIVE_TECHNICAL_HARD_CAP_CNY",
    "QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_SELECTED_COUNT",
    "QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_PROVIDER_CALL_CEILING",
    "QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_FRESH_RESERVATION_CNY",
    "QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_FRESH_STAGE_HARD_CAP_CNY",
    "QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_CUMULATIVE_MAXIMUM_CNY",
    "QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_CUMULATIVE_HARD_CAP_CNY",
    "QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_ACTUAL_COST_CNY",
    "QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_CUMULATIVE_ACTUAL_CNY",
    "QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_RUN_FILE_SHA256",
    "QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_RUN_SHA256",
    "QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_ARTIFACT_SET_SHA256",
    "QWEN38_FEEDBACK_ROUND3_SCHEMA_CANARY_SELECTION_SHA256",
    "QWEN38_FEEDBACK_PHASE60_PREFIX_SELECTED_COUNT",
    "QWEN38_FEEDBACK_PHASE60_PREFIX_PROVIDER_CALL_COUNT",
    "QWEN38_FEEDBACK_PHASE60_PREFIX_RETRY_COUNT",
    "QWEN38_FEEDBACK_PHASE60_NEW_FIRST_CALL_COUNT",
    "QWEN38_FEEDBACK_PHASE60_NEW_RETRY_TOKEN_COUNT",
    "QWEN38_FEEDBACK_PHASE60_NEW_PROVIDER_CALL_CEILING",
    "QWEN38_FEEDBACK_PHASE60_CUMULATIVE_PROVIDER_CALL_CEILING",
    "QWEN38_FEEDBACK_PHASE60_FRESH_MAXIMUM_RESERVATION_CNY",
    "QWEN38_FEEDBACK_PHASE60_FRESH_TECHNICAL_HARD_CAP_CNY",
    "QWEN38_FEEDBACK_PHASE60_CUMULATIVE_MAXIMUM_CNY",
    "QWEN38_FEEDBACK_PHASE60_CUMULATIVE_TECHNICAL_HARD_CAP_CNY",
    "QWEN38_FEEDBACK_THINKING_BUDGET",
    "QWEN38_FEEDBACK_TIMEOUT_SECONDS",
    "QWEN38_FEEDBACK_TRANSPORT_POLICY_SHA256",
    "QWEN38_FEEDBACK_TRANSPORT_POLICY_SHA256_V7",
    "QWEN38_FEEDBACK_TRANSPORT_POLICY_VERSION",
    "QWEN38_FEEDBACK_TRANSPORT_POLICY_VERSION_V7",
    "ROUND3_PRIMARY_TRANSPORT_POLICY_SHA256_V1",
    "ROUND3_PRIMARY_TRANSPORT_POLICY_VERSION_V1",
    "ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_SHA256_V1",
    "ROUND3_PRIMARY_JSON_SCHEMA_TRANSPORT_POLICY_VERSION_V1",
    "ROUND3_PHASE60_RETRY_POLICY_VERSION_V1",
    "ROUND3_PHASE60_RETRY_POLICY_SHA256_V1",
    "Qwen38FeedbackModelSourceLockV1",
    "Qwen38FeedbackModelSourceLockV2",
    "Qwen38FeedbackModelSourceLockV3",
    "Qwen38FeedbackModelSourceLockV4",
    "Qwen38FeedbackModelSourceLockV5",
    "Qwen38FeedbackPricingLockV3",
    "Qwen38FeedbackPricingLockV4",
    "Qwen38FeedbackPricingLockV5",
    "Qwen38FeedbackPricingLockV6",
    "Qwen38FeedbackPricingLockV7",
    "Qwen38FeedbackPricingLockV8",
    "Qwen38FeedbackRoleSelectionV12",
    "Qwen38FeedbackRoleSelectionV13",
    "Qwen38FeedbackRoleSelectionV14",
    "Qwen38FeedbackRoleSelectionV15",
    "Qwen38FeedbackSourceEvidenceV1",
    "SelectedQwenFeedbackAssetV1",
    "build_selected_qwen_feedback_authorization",
    "build_qwen37_feedback_launch_lock",
    "load_qwen37_feedback_launch_lock",
    "load_qwen37_feedback_model_source_lock",
    "load_qwen37_feedback_pricing_lock",
    "load_qwen37_feedback_pricing_lock_v2",
    "load_qwen38_feedback_model_source_lock",
    "load_qwen38_feedback_model_source_lock_v2",
    "load_qwen38_feedback_model_source_lock_v3",
    "load_qwen38_feedback_model_source_lock_v4",
    "load_qwen38_feedback_model_source_lock_v5",
    "load_qwen38_feedback_pricing_lock_v3",
    "load_qwen38_feedback_pricing_lock_v4",
    "load_qwen38_feedback_pricing_lock_v5",
    "load_qwen38_feedback_pricing_lock_v6",
    "load_qwen38_feedback_pricing_lock_v7",
    "load_qwen38_feedback_pricing_lock_v8",
    "load_qwen38_feedback_role_selection_v12",
    "load_qwen38_feedback_role_selection_v13",
    "load_qwen38_feedback_role_selection_v14",
    "load_qwen38_feedback_role_selection_v15",
    "load_selected_qwen_feedback_authorization",
    "require_qwen37_feedback_pre_call_budget",
    "require_qwen37_feedback_pre_call_budget_v2",
    "require_qwen38_feedback_pre_call_budget",
    "require_qwen38_feedback_pre_call_budget_v2",
    "require_qwen38_feedback_pre_call_budget_v3",
    "require_qwen38_feedback_pre_call_budget_v4",
    "require_qwen38_feedback_pre_call_budget_v5",
    "round3_phase60_retry_policy_v1",
    "validate_selected_qwen_feedback_authorization",
    "write_qwen37_feedback_launch_lock",
    "write_selected_qwen_feedback_authorization",
]
