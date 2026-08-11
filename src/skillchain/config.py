"""全局配置：模型角色、端点、路径、论文锚点参数（Table 7）。

所有常量集中于此，其他模块不得自带魔法数字。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
QUERIES_DIR = DATA_DIR / "queries"
RUNS_DIR = ROOT / "runs"
SKILLS_BANK_DIR = ROOT / "skills_bank"
USAGE_LOG = RUNS_DIR / "usage.jsonl"

# ---- 模型角色按用途显式分离：Assistant Qwen / Author Codex /
#      AIFast Gemini visual Feedback / DashScope Kimi final Judge。----
ASSISTANT_PROVIDER = "qwen"
ASSISTANT_MODEL = "qwen3-vl-flash-2026-01-22"
ASSISTANT_MODEL_REVISION = "2026-01-22"

# Backward-compatible names for code that still calls the production Assistant
# the backbone.  Data-label synthesis has separate constants below so changing
# the evaluated Assistant cannot silently change the generated corpus.
BACKBONE_PROVIDER = ASSISTANT_PROVIDER
BACKBONE_MODEL = ASSISTANT_MODEL
LABEL_VISION_SYNTH_PROVIDER = "qwen"
# Historical corpus tooling still uses moving aliases.  It is intentionally
# outside the formal Assistant/Author/Judge locks and is not replay-eligible
# until a separate exact revision and artifact namespace are frozen.
LABEL_VISION_SYNTH_MODEL = "qwen3-vl-plus"

# The prospective static Author runs through the Codex CLI, not llm.chat().
AUTHOR_PROVIDER = "codex_internal"
AUTHOR_MODEL = "gpt-5.6-sol"
AUTHOR_REASONING_EFFORT = "high"

# Historical DashScope authoring artifacts remain reproducible and immutable.
LEGACY_AUTHOR_PROVIDER = "qwen"
LEGACY_AUTHOR_MODEL = "qwen3-vl-flash-2026-01-22"
LEGACY_AUTHOR_MODEL_REVISION = "2026-01-22"
LEGACY_AUTHOR_FALLBACK_MODEL = "qwen3-vl-plus-2025-12-19"
LEGACY_AUTHOR_FALLBACK_MODEL_REVISION = "2025-12-19"

# Historical names are retained because accepted Codex authoring freeze
# builders import them and bind the 2026-07-24 role selection.
JUDGE_PROVIDER = "kimi"
JUDGE_MODEL = "kimi/kimi-k3"
JUDGE_REASONING_EFFORT = "max"
JUDGE_TEMPERATURE = 1.0
JUDGE_TOP_P = 0.95

# Active Portfolio evaluator selection. Qwen3.7 Plus is the DashScope-deployed
# multimodal Feedback model.  Gemini 3.6 Flash remains on the independent
# AIFast final-Judge path.
PORTFOLIO_JUDGE_PROVIDER = "gemini"
PORTFOLIO_JUDGE_MODEL = "gemini-3.6-flash"
# Gemini 3.6 deprecates temperature/top_p/top_k. The AIFast OpenAI-compatible
# route does not expose an attested thinking control, so these controls remain
# omitted from the final-Judge wire.
PORTFOLIO_JUDGE_THINKING = False
PORTFOLIO_JUDGE_THINKING_BUDGET = None
PORTFOLIO_JUDGE_TEMPERATURE = None
PORTFOLIO_JUDGE_TOP_P = None

FEEDBACK_JUDGE_PROVIDER = "qwen"
FEEDBACK_JUDGE_MODEL = "qwen3.7-plus-2026-05-26"
FEEDBACK_JUDGE_MODEL_REVISION = "2026-05-26"
# Freeze the owner-selected thinking Qwen3.7 Plus structured-output path.
# ``enable_thinking=true`` is sent explicitly.  Sampling controls and seed are
# omitted because the selected snapshot's Feedback contract does not freeze
# them; strict shape validation remains local after JSON-object decoding.
FEEDBACK_JUDGE_THINKING = True
FEEDBACK_JUDGE_THINKING_CONTROL = "explicit-enable_thinking-true"
FEEDBACK_JUDGE_THINKING_BUDGET = 2048
FEEDBACK_JUDGE_MAX_TOKENS = None
FEEDBACK_JUDGE_MAX_COMPLETION_TOKENS = 4096
FEEDBACK_JUDGE_TIMEOUT_SECONDS = 600
FEEDBACK_JUDGE_TEMPERATURE = None
FEEDBACK_JUDGE_TOP_P = None

# Exact historical Kimi Feedback controls remain available so immutable v7/v8
# receipts can still be reconstructed and verified after the active role move.
LEGACY_KIMI_FEEDBACK_JUDGE_MODEL = "kimi-k2.6"
LEGACY_KIMI_FEEDBACK_JUDGE_THINKING = False
LEGACY_KIMI_FEEDBACK_JUDGE_TEMPERATURE = 0.6
LEGACY_KIMI_FEEDBACK_JUDGE_TOP_P = 0.95

LEGACY_FEEDBACK_JUDGE_PROVIDER = "deepseek"
LEGACY_FEEDBACK_JUDGE_MODEL = "deepseek-v4-pro"
LABEL_TEXT_REVIEW_PROVIDER = "deepseek"
LABEL_TEXT_REVIEW_MODEL = "deepseek-v4-pro"
# Label review remains a separate moving alias and is not an evaluator.
# Creator/Refiner 无 API：由 Claude Code 会话承担（人机协作批处理，见各 stage 模块说明）。
# llm.chat() 对该 provider 直接报错以防误用；若未来接入 Claude API，切回 "claude"。
REFINER_PROVIDER = "claude_code"
REFINER_MODEL = "claude-sonnet-4-6"  # 仅在接入 Claude API 时生效（对齐论文 Table 7）

# Provider identity 是运行清单的一部分。端点允许通过显式环境变量切换，
# 但 provider、model family 与实际 response model 必须在 llm.py 中再次校验。
DASHSCOPE_BASE_URL = os.environ.get(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
KIMI_DASHSCOPE_BASE_URL = os.environ.get("KIMI_DASHSCOPE_BASE_URL", DASHSCOPE_BASE_URL)
AIFAST_BASE_URL = os.environ.get("AIFAST_BASE_URL", "https://www.aifast.club/v1")
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

PROVIDER_ENDPOINTS = {
    "qwen": DASHSCOPE_BASE_URL,
    "deepseek": DASHSCOPE_BASE_URL,
    "kimi": KIMI_DASHSCOPE_BASE_URL,
    "gemini": AIFAST_BASE_URL,
    "claude": ANTHROPIC_BASE_URL,
}
PROVIDER_API_KEY_ENV = {
    "qwen": "DASHSCOPE_API_KEY",
    "deepseek": "DASHSCOPE_API_KEY",
    "kimi": "DASHSCOPE_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
}
PROVIDER_MODEL_PREFIXES = {
    "qwen": ("qwen",),
    "deepseek": ("deepseek",),
    "kimi": ("kimi-k2.6",),
    "gemini": ("gemini-",),
    "claude": ("claude",),
}
PROVIDER_DEFAULT_MODELS = {
    "qwen": ASSISTANT_MODEL,
    "deepseek": LABEL_TEXT_REVIEW_MODEL,
    "kimi": LEGACY_KIMI_FEEDBACK_JUDGE_MODEL,
    "gemini": PORTFOLIO_JUDGE_MODEL,
    "claude": REFINER_MODEL,
}

# ---- 论文锚点参数（Table 7）----
STAGE2_MAX_ROUNDS = 4
STAGE2_MIN_FAILURES_PER_SKILL = 30
STAGE3_MAX_ROUNDS = 3
STAGE3_MIN_SAMPLES_PER_SKILL = 50
POOR_TIER_THRESHOLDS = {"TCR": 0.20, "CCC": 0.10, "CQ": 0.05, "CA": 0.10}

# ---- 数据规模（访谈决定：对齐论文规模，开发期锁 dev_mini）----
SPLIT_SIZES = {"dev_mini": 200, "opt_pool": 2800, "val": 500, "test_frozen": 1000}
MIN_TEST_PER_INTENT = 150
BOUNDARY_SAMPLE_RATIO = (0.15, 0.20)  # 边界模糊样本占比目标区间

# ---- 助手循环 ----
MAX_TOOL_CALLS_PER_QUERY = 4
