from __future__ import annotations

import base64
from contextlib import nullcontext
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import requests
from PIL import Image

from skillchain.tools.embedding import (
    CachedEmbeddingBackend,
    CacheCorruptionError,
    DashScopeEmbeddingClient,
    EmbeddingCache,
    EmbeddingResponseError,
    FormalEmbeddingBackend,
    ModelDriftError,
    OpenCLIPEmbeddingBackend,
)
from skillchain.tools.model_artifacts import (
    load_model_artifact_manifest,
    publish_model_artifact_manifest,
)
from skillchain.tools.serialization import canonical_json_bytes
from skillchain.tools.settings import (
    DASHSCOPE_EMBEDDING_ENDPOINT,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
)


class FakeResponse:
    def __init__(
        self, status_code: int = 200, payload: dict | None = None, headers=None
    ):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self):
        return self._payload

    @property
    def text(self):
        return str(self._payload)


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, *, headers, json, timeout):
        self.calls.append((url, headers, json, timeout))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def vector(seed: int) -> list[float]:
    values = np.zeros(EMBEDDING_DIMENSION, dtype=np.float64)
    values[seed] = 1.0
    return values.tolist()


def payload(indices=(0, 1), request_id="req-1"):
    return {
        "output": {
            "embeddings": [
                {"index": index, "embedding": vector(index), "type": "vl"}
                for index in indices
            ]
        },
        "usage": {"input_tokens": 2},
        "request_id": request_id,
    }


def _open_clip_artifact(root: Path):
    root.mkdir()
    files = {
        "weights": root / "weights.pt",
        "config": root / "config.json",
        "tokenizer": root / "bpe.txt.gz",
    }
    files["weights"].write_bytes(b"locked-open-clip-weights")
    files["config"].write_bytes(
        canonical_json_bytes(
            {
                "device": "cpu",
                "image_interpolation": "bicubic",
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_resize_mode": "shortest",
                "image_std": [0.26862954, 0.26130258, 0.27577711],
                "model_name": "RN50",
                "output_dimension": 1024,
                "precision": "fp32",
                "pretrained": "openai",
                "schema_version": 1,
                "tokenizer_kind": "simple_bpe",
                "torch_version": "2.7.0",
                "transformers_version": None,
            }
        )
    )
    files["tokenizer"].write_bytes(b"locked-tokenizer")
    published = publish_model_artifact_manifest(
        root / "manifest.json",
        artifact_kind="multimodal_embedding",
        model_id="open-clip:RN50:openai",
        backend_name="open-clip-torch",
        backend_version="3.2.0",
        artifacts=files,
    )
    return load_model_artifact_manifest(
        root / "manifest.json",
        expected_kind="multimodal_embedding",
        expected_manifest_sha256=published.manifest.manifest_sha256,
    )


def _multilingual_open_clip_artifact(root: Path):
    root.mkdir()
    tokenizer_root = root / "tokenizer"
    tokenizer_root.mkdir()
    files = {
        "weights": root / "weights.pt",
        "config": root / "config.json",
        "tokenizer": tokenizer_root / "tokenizer.json",
        "tokenizer_config": tokenizer_root / "tokenizer_config.json",
        "tokenizer_model": tokenizer_root / "sentencepiece.bpe.model",
        "transformer_config": tokenizer_root / "config.json",
    }
    for role, path in files.items():
        if role not in {"config"}:
            path.write_bytes(f"locked-{role}".encode())
    files["config"].write_bytes(
        canonical_json_bytes(
            {
                "device": "cpu",
                "image_interpolation": "bicubic",
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_resize_mode": "shortest",
                "image_std": [0.26862954, 0.26130258, 0.27577711],
                "model_name": "xlm-roberta-large-ViT-H-14",
                "output_dimension": 1024,
                "precision": "fp32",
                "pretrained": "frozen_laion5b_s13b_b90k",
                "schema_version": 1,
                "tokenizer_kind": "hf_local",
                "torch_version": "2.7.0",
                "transformers_version": "5.0.0",
            }
        )
    )
    published = publish_model_artifact_manifest(
        root / "manifest.json",
        artifact_kind="multimodal_embedding",
        model_id=("open-clip:xlm-roberta-large-ViT-H-14:frozen_laion5b_s13b_b90k"),
        backend_name="open-clip-torch",
        backend_version="3.2.0",
        artifacts=files,
    )
    return load_model_artifact_manifest(
        root / "manifest.json",
        expected_kind="multimodal_embedding",
        expected_manifest_sha256=published.manifest.manifest_sha256,
    )


def test_open_clip_backend_is_artifact_locked_local_and_supports_bytes(
    tmp_path: Path,
    monkeypatch,
):
    artifact = _open_clip_artifact(tmp_path / "embedding")

    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class Model:
        def eval(self):
            return self

        def encode_text(self, tokens):
            values = np.zeros((len(tokens), EMBEDDING_DIMENSION), dtype=np.float32)
            values[:, 0] = 1.0
            return Tensor(values)

        def encode_image(self, images):
            values = np.zeros((len(images), EMBEDDING_DIMENSION), dtype=np.float32)
            values[:, 1] = 1.0
            return Tensor(values)

    class Tokenizer:
        def __init__(self, *, bpe_path):
            assert bpe_path.endswith("bpe.txt.gz")

        def __call__(self, texts):
            return list(texts)

    open_clip = ModuleType("open_clip")
    open_clip.create_model_and_transforms = lambda *_args, **_kwargs: (
        Model(),
        object(),
        lambda image: image,
    )
    open_clip.SimpleTokenizer = Tokenizer
    torch = ModuleType("torch")
    torch.inference_mode = nullcontext
    torch.stack = lambda values: list(values)
    monkeypatch.setitem(sys.modules, "open_clip", open_clip)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        "skillchain.tools.embedding.metadata.version",
        lambda name: {"open_clip_torch": "3.2.0", "torch": "2.7.0"}[name],
    )

    image = tmp_path / "query.png"
    Image.new("RGB", (4, 4), "red").save(image)
    content = image.read_bytes()
    backend = OpenCLIPEmbeddingBackend(artifact)

    assert backend.model == "open-clip:RN50:openai"
    assert backend.execution_location == "local"
    assert len(backend.formal_runtime_binding_sha256) == 64
    assert backend.embed_texts(["商品"]).shape == (1, EMBEDDING_DIMENSION)
    assert backend.embed_image_bytes([content]).shape == (1, EMBEDDING_DIMENSION)


def test_multilingual_open_clip_uses_only_locked_local_tokenizer(
    tmp_path: Path,
    monkeypatch,
):
    artifact = _multilingual_open_clip_artifact(tmp_path / "embedding")
    observed = {}

    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class Model:
        def eval(self):
            return self

        def encode_text(self, tokens):
            values = np.zeros((len(tokens), EMBEDDING_DIMENSION), dtype=np.float32)
            values[:, 0] = 1.0
            return Tensor(values)

    class HFTokenizer:
        def __init__(self, path, **kwargs):
            observed["tokenizer"] = (Path(path), kwargs)

        def __call__(self, texts):
            return list(texts)

    open_clip = ModuleType("open_clip")
    open_clip.get_model_config = lambda _name: {"text_cfg": {"remote": "blocked"}}

    def create_model_and_transforms(*_args, **kwargs):
        observed["model"] = kwargs
        return Model(), object(), object()

    open_clip.create_model_and_transforms = create_model_and_transforms
    tokenizer_module = ModuleType("open_clip.tokenizer")
    tokenizer_module.HFTokenizer = HFTokenizer
    torch = ModuleType("torch")
    torch.inference_mode = nullcontext
    monkeypatch.setitem(sys.modules, "open_clip", open_clip)
    monkeypatch.setitem(sys.modules, "open_clip.tokenizer", tokenizer_module)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(
        "skillchain.tools.embedding.metadata.version",
        lambda name: {
            "open_clip_torch": "3.2.0",
            "torch": "2.7.0",
            "transformers": "5.0.0",
        }[name],
    )

    backend = OpenCLIPEmbeddingBackend(artifact)
    assert backend.embed_texts(["中文商品查询"]).shape == (1, EMBEDDING_DIMENSION)
    tokenizer_root = artifact.artifact_path("tokenizer").parent
    assert observed["tokenizer"] == (
        tokenizer_root,
        {"local_files_only": True},
    )
    assert observed["model"]["pretrained_text"] is False
    assert observed["model"]["text_cfg"]["hf_model_name"] == str(tokenizer_root)


def test_formal_dashscope_binding_accepts_only_authority_owned_default_session():
    client = DashScopeEmbeddingClient(api_key="secret")
    second = DashScopeEmbeddingClient(api_key="other-secret")

    assert type(client._session) is requests.Session
    assert client._session.trust_env is False
    assert len(client.formal_runtime_binding_sha256) == 64
    assert client.formal_runtime_binding_sha256 == second.formal_runtime_binding_sha256

    external = requests.Session()
    external.trust_env = False
    explicit = DashScopeEmbeddingClient(api_key="secret", session=external)
    with pytest.raises(ValueError, match="authority-owned default Session"):
        _ = explicit.formal_runtime_binding_sha256


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        (
            lambda session: session.mount("https://", requests.adapters.HTTPAdapter()),
            "objects",
        ),
        (
            lambda session: session.adapters.__setitem__(
                "https://", requests.adapters.HTTPAdapter()
            ),
            "objects",
        ),
        (
            lambda session: session.hooks["response"].append(lambda response: response),
            "hooks",
        ),
        (
            lambda session: session.proxies.update({"https": "http://proxy.test"}),
            "proxies",
        ),
        (lambda session: session.cookies.set("sid", "attacker"), "cookies"),
        (
            lambda session: setattr(
                session.cookies.get_policy(), "strict_domain", True
            ),
            "cookie policy",
        ),
        (lambda session: setattr(session, "verify", False), "TLS policy"),
        (lambda session: setattr(session, "cert", "client.pem"), "TLS policy"),
        (lambda session: session.headers.update({"X-Test": "changed"}), "headers"),
        (
            lambda session: setattr(
                session.adapters["https://"],
                "max_retries",
                requests.adapters.Retry(total=3),
            ),
            "adapter policy",
        ),
        (
            lambda session: session.adapters[
                "https://"
            ].poolmanager.connection_pool_kw.update({"ssl_context": object()}),
            "connection pool policy",
        ),
        (
            lambda session: setattr(session, "post", lambda *_args, **_kwargs: None),
            "unreviewed",
        ),
    ),
)
def test_formal_dashscope_binding_rejects_mutable_network_state(mutation, expected):
    client = DashScopeEmbeddingClient(api_key="secret")
    assert len(client.formal_runtime_binding_sha256) == 64

    mutation(client._session)

    with pytest.raises(ValueError, match=expected):
        _ = client.formal_runtime_binding_sha256


def test_authority_rechecks_default_session_after_http_send(monkeypatch):
    client = DashScopeEmbeddingClient(api_key="secret")

    def send(_adapter, request, **_kwargs):
        client._session.verify = False
        response = requests.Response()
        response.status_code = 200
        response.request = request
        response._content = json.dumps(payload((0,))).encode("utf-8")
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)

    with pytest.raises(ValueError, match="TLS policy"):
        client.embed_texts(["query"])


def test_dashscope_batches_and_restores_response_indices():
    session = FakeSession([FakeResponse(payload=payload((1, 0)))])
    client = DashScopeEmbeddingClient(
        api_key="secret", session=session, sleeper=lambda _: None
    )

    result = client.embed_texts(["first", "second"])

    assert result.shape == (2, EMBEDDING_DIMENSION)
    assert result.dtype == np.float32
    assert np.allclose(result[0], np.eye(EMBEDDING_DIMENSION)[0])
    assert np.allclose(result[1], np.eye(EMBEDDING_DIMENSION)[1])
    url, headers, request, _ = session.calls[0]
    assert url == DASHSCOPE_EMBEDDING_ENDPOINT
    assert headers["Authorization"] == "Bearer secret"
    assert request["model"] == EMBEDDING_MODEL
    assert request["parameters"] == {"dimension": EMBEDDING_DIMENSION}
    assert "enable_fusion" not in request["parameters"]
    assert [c["text"] for c in request["input"]["contents"]] == ["first", "second"]


@pytest.mark.parametrize("text", ["", "   ", "\t\n"])
def test_dashscope_rejects_blank_text(text):
    client = DashScopeEmbeddingClient(
        api_key="secret", session=FakeSession([]), sleeper=lambda _: None
    )
    with pytest.raises(ValueError, match="blank"):
        client.embed_texts([text])


def test_dashscope_rejects_malformed_response_without_request_id():
    session = FakeSession([FakeResponse(payload={"output": {"embeddings": []}})])
    client = DashScopeEmbeddingClient(
        api_key="secret", session=session, sleeper=lambda _: None
    )
    with pytest.raises(EmbeddingResponseError, match="request_id"):
        client.embed_texts(["query"])


@pytest.mark.parametrize("output", [None, [], "not-an-object"])
def test_dashscope_rejects_malformed_output_shapes(output):
    session = FakeSession(
        [FakeResponse(payload={"output": output, "request_id": "req"})]
    )
    client = DashScopeEmbeddingClient(
        api_key="secret", session=session, sleeper=lambda _: None
    )
    with pytest.raises(EmbeddingResponseError):
        client.embed_texts(["query"])


def test_dashscope_retries_timeout_and_connection_errors():
    session = FakeSession(
        [
            requests.Timeout("timed out"),
            ConnectionError("connection reset"),
            FakeResponse(payload=payload((0,), request_id="req-ok")),
        ]
    )
    client = DashScopeEmbeddingClient(
        api_key="secret", session=session, sleeper=lambda _: None
    )

    assert client.embed_texts(["query"]).shape == (1, EMBEDDING_DIMENSION)
    assert len(session.calls) == 3


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_dashscope_fails_immediately_on_client_errors(status):
    session = FakeSession([FakeResponse(status_code=status)])
    client = DashScopeEmbeddingClient(
        api_key="secret", session=session, sleeper=lambda _: None
    )
    with pytest.raises(Exception):
        client.embed_texts(["query"])
    assert len(session.calls) == 1


def test_dashscope_retries_429_and_5xx_with_retry_after():
    sleeps = []
    session = FakeSession(
        [
            FakeResponse(status_code=429, headers={"Retry-After": "0.25"}),
            FakeResponse(status_code=503),
            FakeResponse(payload=payload((0,), request_id="req-3")),
        ]
    )
    client = DashScopeEmbeddingClient(
        api_key="secret", session=session, sleeper=sleeps.append, backoff_base=0.01
    )

    result = client.embed_texts(["query"])

    assert result.shape == (1, EMBEDDING_DIMENSION)
    assert len(session.calls) == 3
    assert sleeps[0] == pytest.approx(0.25)
    assert sleeps[1] >= 0


def test_dashscope_retries_all_5xx_statuses():
    session = FakeSession(
        [
            FakeResponse(status_code=501),
            FakeResponse(status_code=599),
            FakeResponse(payload=payload((0,), request_id="req-5xx")),
        ]
    )
    client = DashScopeEmbeddingClient(
        api_key="secret", session=session, sleeper=lambda _: None
    )

    assert client.embed_texts(["query"]).shape == (1, EMBEDDING_DIMENSION)
    assert len(session.calls) == 3


def test_dashscope_image_is_data_uri_and_batches_at_five(tmp_path: Path):
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (2, 2), (1, 2, 3)).save(image_path, format="PNG")
    session = FakeSession(
        [
            FakeResponse(payload=payload((0, 1, 2, 3, 4), request_id="req-0")),
            FakeResponse(payload=payload((0,), request_id="req-1")),
        ]
    )
    client = DashScopeEmbeddingClient(
        api_key="secret", session=session, sleeper=lambda _: None
    )

    result = client.embed_images([image_path] * 6)

    assert result.shape == (6, EMBEDDING_DIMENSION)
    assert len(session.calls) == 2
    image_value = session.calls[0][2]["input"]["contents"][0]["image"]
    assert image_value.startswith("data:image/png;base64,")
    assert base64.b64decode(image_value.split(",", 1)[1]) == image_path.read_bytes()


def test_dashscope_allows_image_exactly_ten_mib(tmp_path: Path):
    image_path = tmp_path / "exact.png"
    Image.new("RGB", (1, 1), (1, 2, 3)).save(image_path, format="PNG")
    image_path.write_bytes(
        image_path.read_bytes() + b"\0" * (10 * 1024 * 1024 - image_path.stat().st_size)
    )
    session = FakeSession([FakeResponse(payload=payload((0,), request_id="req-exact"))])
    client = DashScopeEmbeddingClient(
        api_key="secret", session=session, sleeper=lambda _: None
    )

    assert client.embed_images([image_path]).shape == (1, EMBEDDING_DIMENSION)


def test_cache_is_wal_and_round_trips_normalized_vectors(tmp_path: Path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    value = np.zeros(EMBEDDING_DIMENSION, dtype=np.float32)
    value[0] = 1
    cache.put_many("text", [("hello", value, "req-1")])

    assert cache.get("text", "hello") is not None
    with sqlite3.connect(tmp_path / "cache.sqlite3") as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(embeddings)")
        }
        assert {"model", "dimension", "modality", "input_sha256"} <= columns


def test_cached_backend_only_calls_missing_inputs_and_resumes(tmp_path: Path):
    class FakeBackend:
        def __init__(self):
            self.calls = []

        def embed_texts(self, values):
            self.calls.append(list(values))
            return np.tile(
                np.eye(EMBEDDING_DIMENSION, dtype=np.float32)[0], (len(values), 1)
            )

    backend = FakeBackend()
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    cached = CachedEmbeddingBackend(backend, cache)
    cached.embed_texts(["a", "b"])
    cached.embed_texts(["a", "b", "c"])

    assert backend.calls == [["a", "b"], ["c"]]


def test_cached_backend_deduplicates_and_commits_each_batch(tmp_path: Path):
    class FailingBackend:
        def __init__(self):
            self.calls = []

        def embed_texts(self, values):
            self.calls.append(list(values))
            if len(self.calls) == 2:
                raise RuntimeError("interrupted after first batch")
            return np.tile(
                np.eye(EMBEDDING_DIMENSION, dtype=np.float32)[0], (len(values), 1)
            )

    values = [f"text-{index}" for index in range(20)] + [
        "text-20",
        "text-20",
        "text-21",
    ]
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    backend = FailingBackend()
    cached = CachedEmbeddingBackend(backend, cache)
    with pytest.raises(RuntimeError, match="interrupted"):
        cached.embed_texts(values)
    assert backend.calls == [values[:20], ["text-20", "text-21"]]
    assert cache.get("text", "text-0") is not None
    assert cache.get("text", "text-20") is None

    class ResumingBackend:
        def __init__(self):
            self.calls = []

        def embed_texts(self, values):
            self.calls.append(list(values))
            return np.tile(
                np.eye(EMBEDDING_DIMENSION, dtype=np.float32)[1], (len(values), 1)
            )

    resuming = ResumingBackend()
    resumed = CachedEmbeddingBackend(resuming, cache)
    result = resumed.embed_texts(values)
    assert resuming.calls == [["text-20", "text-21"]]
    assert result.shape == (len(values), EMBEDDING_DIMENSION)
    assert np.array_equal(result[21], result[22])


def test_cached_backend_counts_real_requests_and_cache_hit_input_vectors(
    tmp_path: Path,
):
    class Backend:
        def __init__(self):
            self.calls = []

        def embed_texts(self, values):
            self.calls.append(list(values))
            return np.tile(
                np.eye(EMBEDDING_DIMENSION, dtype=np.float32)[0], (len(values), 1)
            )

    backend = Backend()
    cached = CachedEmbeddingBackend(
        backend,
        EmbeddingCache(tmp_path / "cache.sqlite3"),
        canary_text="fixed-canary",
        canary_vector=np.eye(EMBEDDING_DIMENSION, dtype=np.float32)[0],
    )

    cached.embed_texts(["first", "first"])
    cached.embed_texts(["first", "second"])

    assert backend.calls == [["fixed-canary"], ["first"], ["second"]]
    assert cached.api_calls == 3
    assert cached.cache_hits == 1
    assert cached.api_calls_by_modality == {"image": 0, "text": 3}
    assert cached.cache_hits_by_modality == {"image": 0, "text": 1}


def test_establish_canary_clears_old_model_cache_only_after_a_successful_response(
    tmp_path: Path,
):
    image_path = tmp_path / "old.png"
    Image.new("RGB", (1, 1), (1, 2, 3)).save(image_path)
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    old_vector = np.eye(EMBEDDING_DIMENSION, dtype=np.float32)[0]
    cache.put_many(
        "text",
        [
            ("old-product", old_vector, "old-text"),
            ("old-canary", old_vector, "old-canary"),
        ],
    )
    cache.put_many("image", [(image_path, old_vector, "old-image")])

    class NewModelBackend:
        def embed_texts(self, values):
            vector = np.eye(EMBEDDING_DIMENSION, dtype=np.float32)[1]
            return np.tile(vector, (len(values), 1))

    cached = CachedEmbeddingBackend(NewModelBackend(), cache)
    cached.establish_canary("fresh-canary")

    assert cache.get("text", "old-product") is None
    assert cache.get("text", "old-canary") is None
    assert cache.get("image", image_path) is None
    assert cache.get("text", "fresh-canary") is not None

    class FailingBackend:
        def embed_texts(self, values):
            raise RuntimeError("canary request failed")

    cache.put_many("text", [("preserved", old_vector, "old")])
    failed = CachedEmbeddingBackend(FailingBackend(), cache)
    with pytest.raises(RuntimeError, match="canary request failed"):
        failed.establish_canary("will-not-replace")
    assert cache.get("text", "preserved") is not None


def test_establish_canary_rolls_back_cache_deletion_when_canary_insert_fails(
    tmp_path: Path,
):
    image_path = tmp_path / "old.png"
    Image.new("RGB", (1, 1), (1, 2, 3)).save(image_path)
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    old_vector = np.eye(EMBEDDING_DIMENSION, dtype=np.float32)[0]
    cache.put_many("text", [("old-product", old_vector, "old-text")])
    cache.put_many("image", [(image_path, old_vector, "old-image")])
    canary_text = "fresh-canary"
    digest = hashlib.sha256(canary_text.encode("utf-8")).hexdigest()
    with sqlite3.connect(tmp_path / "cache.sqlite3") as connection:
        connection.execute(
            f"""
            CREATE TRIGGER fail_fresh_canary_insert
            BEFORE INSERT ON embeddings
            WHEN NEW.input_sha256 = '{digest}'
            BEGIN
                SELECT RAISE(ABORT, 'fresh canary insert blocked');
            END
            """
        )

    class NewModelBackend:
        def embed_texts(self, values):
            vector = np.eye(EMBEDDING_DIMENSION, dtype=np.float32)[1]
            return np.tile(vector, (len(values), 1))

    cached = CachedEmbeddingBackend(NewModelBackend(), cache)
    with pytest.raises(sqlite3.IntegrityError, match="fresh canary insert blocked"):
        cached.establish_canary(canary_text)

    assert cache.get("text", "old-product") is not None
    assert cache.get("image", image_path) is not None
    assert cached.canary_text is None
    assert cached.canary_vector is None


def test_establish_canary_returns_a_defensive_vector_copy(tmp_path: Path):
    class Backend:
        def embed_texts(self, values):
            vector = np.eye(EMBEDDING_DIMENSION, dtype=np.float32)[1]
            return np.tile(vector, (len(values), 1))

    cached = CachedEmbeddingBackend(
        Backend(), EmbeddingCache(tmp_path / "cache.sqlite3")
    )

    result = cached.establish_canary("fresh-canary")
    result[0, 1] = 0.0

    expected = np.eye(EMBEDDING_DIMENSION, dtype=np.float32)[1]
    assert np.array_equal(cached.canary_vector, expected)


def test_cached_backend_persists_dashscope_request_id(tmp_path: Path):
    session = FakeSession(
        [FakeResponse(payload=payload((0,), request_id="dash-req-7"))]
    )
    client = DashScopeEmbeddingClient(
        api_key="secret", session=session, sleeper=lambda _: None
    )
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    CachedEmbeddingBackend(client, cache).embed_texts(["query"])

    digest = hashlib.sha256(b"query").hexdigest()
    with sqlite3.connect(tmp_path / "cache.sqlite3") as connection:
        request_id = connection.execute(
            "SELECT request_id FROM embeddings WHERE input_sha256 = ?", (digest,)
        ).fetchone()[0]
    assert request_id == "dash-req-7"


def test_corrupt_cache_blob_is_rejected(tmp_path: Path):
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    digest = hashlib.sha256(b"hello").hexdigest()
    with sqlite3.connect(tmp_path / "cache.sqlite3") as connection:
        connection.execute(
            "INSERT INTO embeddings VALUES (?, ?, ?, ?, ?, ?, ?)",
            (EMBEDDING_MODEL, EMBEDDING_DIMENSION, "text", digest, 5, b"bad", "req"),
        )
    with pytest.raises(CacheCorruptionError):
        cache.get("text", "hello")


def test_canary_drift_rejects_uncached_query_without_writing(tmp_path: Path):
    class FakeBackend:
        def __init__(self):
            self.calls = []

        def embed_texts(self, values):
            self.calls.append(list(values))
            return np.tile(
                np.eye(EMBEDDING_DIMENSION, dtype=np.float32)[1], (len(values), 1)
            )

    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    backend = FakeBackend()
    cached = CachedEmbeddingBackend(
        backend,
        cache,
        canary_text="fixed-canary",
        canary_vector=np.eye(EMBEDDING_DIMENSION, dtype=np.float32)[0],
    )
    with pytest.raises(ModelDriftError):
        cached.embed_texts(["query"])
    assert cache.get("text", "query") is None


def test_canary_check_is_skipped_when_query_is_fully_cached(tmp_path: Path):
    class OfflineBackend:
        def embed_texts(self, values):
            raise AssertionError("cache hit must not use backend")

    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    value = np.eye(EMBEDDING_DIMENSION, dtype=np.float32)[0]
    cache.put_many("text", [("query", value, "req")])
    cached = CachedEmbeddingBackend(
        OfflineBackend(),
        cache,
        canary_text="fixed-canary",
        canary_vector=value,
    )
    assert cached.embed_texts(["query"]).shape == (1, EMBEDDING_DIMENSION)


def test_canary_is_checked_once_per_backend_instance(tmp_path: Path):
    class Backend:
        def __init__(self):
            self.calls = []

        def embed_texts(self, values):
            self.calls.append(list(values))
            return np.tile(
                np.eye(EMBEDDING_DIMENSION, dtype=np.float32)[0], (len(values), 1)
            )

    value = np.eye(EMBEDDING_DIMENSION, dtype=np.float32)[0]
    backend = Backend()
    cached = CachedEmbeddingBackend(
        backend,
        EmbeddingCache(tmp_path / "cache.sqlite3"),
        canary_text="fixed-canary",
        canary_vector=value,
    )

    cached.embed_texts(["first"])
    cached.embed_texts(["second"])

    assert backend.calls == [["fixed-canary"], ["first"], ["second"]]


def test_cached_backend_rejects_blank_text_before_cache_hit(tmp_path: Path):
    value = np.eye(EMBEDDING_DIMENSION, dtype=np.float32)[0]
    cache = EmbeddingCache(tmp_path / "cache.sqlite3")
    cache.put_many("text", [("   ", value, "req")])

    class OfflineBackend:
        def embed_texts(self, values):
            raise AssertionError("blank input must be rejected before backend")

    cached = CachedEmbeddingBackend(OfflineBackend(), cache)
    with pytest.raises(ValueError, match="blank"):
        cached.embed_texts(["   "])


def test_cached_backend_rejects_blank_text_before_fake_backend(tmp_path: Path):
    class Backend:
        def __init__(self):
            self.calls = []

        def embed_texts(self, values):
            self.calls.append(list(values))
            return np.ones((len(values), EMBEDDING_DIMENSION), dtype=np.float32)

    backend = Backend()
    cached = CachedEmbeddingBackend(backend, EmbeddingCache(tmp_path / "cache.sqlite3"))
    with pytest.raises(ValueError, match="blank"):
        cached.embed_texts(["   "])
    assert backend.calls == []


def test_cached_backend_rejects_blank_canary_text(tmp_path: Path):
    value = np.eye(EMBEDDING_DIMENSION, dtype=np.float32)[0]
    with pytest.raises(ValueError, match="blank"):
        CachedEmbeddingBackend(
            object(),
            EmbeddingCache(tmp_path / "cache.sqlite3"),
            canary_text="   ",
            canary_vector=value,
        )


def test_formal_backend_ignores_legal_normalized_sqlite_cache_poison(
    tmp_path: Path,
):
    canary = np.eye(EMBEDDING_DIMENSION, dtype=np.float32)[0]
    poison = np.eye(EMBEDDING_DIMENSION, dtype=np.float32)[1]
    expected = np.eye(EMBEDDING_DIMENSION, dtype=np.float32)[2]
    cache = EmbeddingCache(tmp_path / "attacker-writable.sqlite3")
    cache.put_many("text", [("query", poison, "attacker")])

    class Backend:
        model = EMBEDDING_MODEL
        execution_location = "local"

        @property
        def formal_runtime_binding_sha256(self):
            return hashlib.sha256(b"offline-formal-backend-fixture-v1").hexdigest()

        def __init__(self):
            self.calls = []

        def embed_texts(self, values):
            self.calls.append(list(values))
            vector = canary if values == ["fixed-canary"] else expected
            return np.tile(vector, (len(values), 1))

    backend = Backend()
    formal = FormalEmbeddingBackend(
        backend,
        canary_text="fixed-canary",
        canary_vector=canary,
    )
    formal._canary_verified = True  # type: ignore[attr-defined]  # noqa: SLF001
    with pytest.raises(ModelDriftError, match="has not been verified"):
        formal.embed_texts(["query"])
    formal.verify_canary(force=True)
    result = formal.embed_texts(["query"])

    assert np.array_equal(cache.get("text", "query"), poison)
    assert np.array_equal(result[0], expected)
    assert backend.calls == [["fixed-canary"], ["query"]]
    assert formal.cache_policy == "bypass"
    assert len(formal.runtime_binding_sha256) == 64


def test_ragged_backend_matrix_is_embedding_response_error(tmp_path: Path):
    class Backend:
        def embed_texts(self, values):
            return [[1], [2, 3]]

    cached = CachedEmbeddingBackend(
        Backend(), EmbeddingCache(tmp_path / "cache.sqlite3")
    )
    with pytest.raises(EmbeddingResponseError):
        cached.embed_texts(["first", "second"])
