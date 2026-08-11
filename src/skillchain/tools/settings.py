"""Pinned configuration and artifact paths for the Phase 2 tool layer."""

from skillchain.config import DATA_DIR, RUNS_DIR

EMBEDDING_MODEL = "qwen3-vl-embedding"
EMBEDDING_DIMENSION = 1024
IMAGE_BATCH_SIZE = 5
TEXT_BATCH_SIZE = 20
PRODUCT_K = 10
KB_K = 5
MMR_LAMBDA = 0.5

DASHSCOPE_EMBEDDING_ENDPOINT = (
    "https://dashscope.aliyuncs.com/api/v1/services/embeddings/"
    "multimodal-embedding/multimodal-embedding"
)

PRODUCTS_SOURCE_PATH = DATA_DIR / "clean" / "products.parquet"
PRODUCT_INDEX_DIR = DATA_DIR / "index" / "products"
PRODUCT_QUERY_ARTIFACT_PATH = DATA_DIR / "queries" / "split_assignment.jsonl"
KB_SOURCE_DIR = DATA_DIR / "kb"
KB_INDEX_DIR = DATA_DIR / "index" / "kb"
KB_CATALOG_DIR = DATA_DIR / "catalog" / "kb"
EMBEDDING_CACHE_PATH = DATA_DIR / "index" / "embedding_cache.sqlite3"
DETECTOR_MODEL_PATH = DATA_DIR / "models" / "yolo11n.pt"
DETECTOR_MANIFEST_PATH = DATA_DIR / "models" / "detector-manifest.json"
OCR_MODEL_DIR = DATA_DIR / "models" / "ocr"
OCR_MANIFEST_PATH = DATA_DIR / "models" / "ocr-manifest.json"
TOOL_REGISTRY_MANIFEST_PATH = DATA_DIR / "index" / "tool-registry.json"
RETRIEVAL_RUNS_DIR = RUNS_DIR / "retrieval"
GOLD_RESULTS_PATH = RUNS_DIR / "phase2" / "gold_results.json"
