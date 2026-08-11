"""DashScope embeddings with a resumable, validated SQLite cache.

The remote service is deliberately kept behind a small protocol so index builders
can be tested with deterministic local backends.  No request payload is logged:
both the authorization header and image data URIs can contain sensitive data.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import partial
import hashlib
from importlib import metadata
from io import BytesIO
import json
import math
import os
import sqlite3
from threading import RLock
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol
from weakref import WeakKeyDictionary

import numpy as np
import requests
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from skillchain.tools.model_artifacts import (
    VerifiedModelArtifact,
)
from skillchain.tools.serialization import canonical_json_bytes, sha256_bytes
from skillchain.tools.settings import (
    DASHSCOPE_EMBEDDING_ENDPOINT,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL,
    IMAGE_BATCH_SIZE,
    TEXT_BATCH_SIZE,
)

_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_MAX_ATTEMPTS = 4
_CANARY_MIN_COSINE = 0.999999
_IMAGE_MEDIA_TYPES = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
    "BMP": "image/bmp",
}
_FORMAL_CANARY_RECEIPTS: WeakKeyDictionary[object, str] = WeakKeyDictionary()
_FORMAL_DASHSCOPE_ENDPOINT = DASHSCOPE_EMBEDDING_ENDPOINT
_REQUESTS_VERSION = requests.__version__
_REQUESTS_SESSION_TYPE = requests.Session
_REQUESTS_ADAPTER_TYPE = requests.adapters.HTTPAdapter
_DEFAULT_SESSION_HEADERS = tuple(sorted(requests.utils.default_headers().items()))
_DEFAULT_COOKIE_POLICY_STATE = tuple(
    sorted(vars(requests.cookies.RequestsCookieJar().get_policy()).items())
)
_FORMAL_SESSION_ISSUER = object()


class EmbeddingError(RuntimeError):
    """Base class for recoverable embedding failures."""


class EmbeddingRequestError(EmbeddingError):
    """The DashScope service could not accept or complete a request."""


class EmbeddingResponseError(EmbeddingError):
    """DashScope returned a response that cannot safely be used."""


class CacheCorruptionError(EmbeddingError):
    """A cached vector no longer satisfies its integrity constraints."""


class ModelDriftError(EmbeddingError):
    """The remote model no longer matches the persisted canary vector."""


class LocalEmbeddingRuntimeUnavailableError(EmbeddingError):
    """The pinned local embedding runtime cannot be loaded."""


@dataclass(frozen=True, slots=True)
class _FormalSessionReceipt:
    session: object
    state_objects: tuple[object, ...]
    policy_sha256: str


class _FormalSessionAuthority:
    """Attest only internally-created, pristine requests sessions."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._records: WeakKeyDictionary[object, _FormalSessionReceipt] = (
            WeakKeyDictionary()
        )

    def issue(
        self,
        client: object,
        session: object,
        *,
        issuer: object,
    ) -> None:
        if issuer is not _FORMAL_SESSION_ISSUER:
            raise ValueError("formal Session authority rejected the issuer")
        policy_sha256, state_objects = _capture_pristine_session(session)
        receipt = _FormalSessionReceipt(
            session=session,
            state_objects=state_objects,
            policy_sha256=policy_sha256,
        )
        with self._lock:
            if client in self._records:
                raise ValueError("formal Session receipt was already issued")
            self._records[client] = receipt

    def validate(self, client: object, session: object) -> str:
        with self._lock:
            receipt = self._records.get(client)
        if receipt is None or receipt.session is not session:
            raise ValueError(
                "formal DashScope runtime requires its authority-owned default Session"
            )
        policy_sha256, state_objects = _capture_pristine_session(session)
        if policy_sha256 != receipt.policy_sha256 or len(state_objects) != len(
            receipt.state_objects
        ):
            raise ValueError("formal DashScope Session state changed after attestation")
        if any(
            current is not expected
            for current, expected in zip(
                state_objects, receipt.state_objects, strict=True
            )
        ):
            raise ValueError(
                "formal DashScope Session objects changed after attestation"
            )
        return policy_sha256

    def validate_if_issued(self, client: object, session: object) -> None:
        with self._lock:
            issued = client in self._records
        if issued:
            self.validate(client, session)


_FORMAL_SESSION_AUTHORITY = _FormalSessionAuthority()


def _pool_key_function_identity(function: object) -> dict[str, object]:
    if not isinstance(function, partial):
        raise ValueError("formal DashScope connection key function changed")
    if function.keywords:
        raise ValueError("formal DashScope connection key options changed")
    if len(function.args) != 1 or not isinstance(function.args[0], type):
        raise ValueError("formal DashScope connection key arguments changed")
    argument = function.args[0]
    return {
        "argument": f"{argument.__module__}.{argument.__qualname__}",
        "function": (f"{function.func.__module__}.{function.func.__qualname__}"),
    }


def _capture_pristine_session(session: object) -> tuple[str, tuple[object, ...]]:
    """Validate the complete reviewed Session policy and return identity locks."""

    if type(session) is not _REQUESTS_SESSION_TYPE:
        raise ValueError("formal DashScope runtime requires the default Session type")
    state = vars(session)
    expected_fields = {
        "adapters",
        "auth",
        "cert",
        "cookies",
        "headers",
        "hooks",
        "max_redirects",
        "params",
        "proxies",
        "stream",
        "trust_env",
        "verify",
    }
    if set(state) != expected_fields:
        raise ValueError("formal DashScope Session has unreviewed instance state")
    if tuple(sorted(session.headers.items())) != _DEFAULT_SESSION_HEADERS:
        raise ValueError("formal DashScope Session headers changed")
    if session.auth is not None:
        raise ValueError("formal DashScope Session auth must be request-scoped")
    if session.proxies != {}:
        raise ValueError("formal DashScope Session proxies must be empty")
    if session.hooks != {"response": []}:
        raise ValueError("formal DashScope Session hooks must be empty")
    if session.params != {}:
        raise ValueError("formal DashScope Session params must be empty")
    if session.stream is not False:
        raise ValueError("formal DashScope Session streaming must be disabled")
    if session.verify is not True or session.cert is not None:
        raise ValueError("formal DashScope Session TLS policy changed")
    if session.max_redirects != requests.models.DEFAULT_REDIRECT_LIMIT:
        raise ValueError("formal DashScope Session redirect policy changed")
    if session.trust_env is not False:
        raise ValueError("formal DashScope Session must ignore environment proxies")
    if type(session.cookies) is not requests.cookies.RequestsCookieJar:
        raise ValueError("formal DashScope Session cookie jar type changed")
    if len(session.cookies) != 0:
        raise ValueError("formal DashScope Session cookies must be empty")
    if (
        tuple(sorted(vars(session.cookies.get_policy()).items()))
        != _DEFAULT_COOKIE_POLICY_STATE
    ):
        raise ValueError("formal DashScope Session cookie policy changed")
    if tuple(session.adapters) != ("https://", "http://"):
        raise ValueError("formal DashScope Session adapter routes changed")

    state_objects: list[object] = [
        session.headers,
        session.proxies,
        session.hooks,
        session.hooks["response"],
        session.params,
        session.cookies,
        session.cookies.get_policy(),
        session.cookies._cookies,
        session.cookies._cookies_lock,
        session.adapters,
    ]
    adapter_payloads: list[dict[str, object]] = []
    for route, adapter in session.adapters.items():
        if type(adapter) is not _REQUESTS_ADAPTER_TYPE:
            raise ValueError("formal DashScope Session requires default HTTP adapters")
        adapter_state = vars(adapter)
        if set(adapter_state) != {
            "_pool_block",
            "_pool_connections",
            "_pool_maxsize",
            "config",
            "max_retries",
            "poolmanager",
            "proxy_manager",
        }:
            raise ValueError("formal DashScope adapter has unreviewed instance state")
        if (
            adapter._pool_connections != requests.adapters.DEFAULT_POOLSIZE
            or adapter._pool_maxsize != requests.adapters.DEFAULT_POOLSIZE
            or adapter._pool_block is not requests.adapters.DEFAULT_POOLBLOCK
            or adapter.config != {}
            or adapter.proxy_manager != {}
            or repr(adapter.max_retries) != repr(requests.adapters.Retry(0, read=False))
        ):
            raise ValueError("formal DashScope HTTP adapter policy changed")
        poolmanager = adapter.poolmanager
        if poolmanager.headers != {} or poolmanager.connection_pool_kw != {
            "maxsize": requests.adapters.DEFAULT_POOLSIZE,
            "block": requests.adapters.DEFAULT_POOLBLOCK,
        }:
            raise ValueError("formal DashScope connection pool policy changed")
        pool_classes = tuple(
            sorted(
                (
                    scheme,
                    f"{pool_type.__module__}.{pool_type.__qualname__}",
                )
                for scheme, pool_type in poolmanager.pool_classes_by_scheme.items()
            )
        )
        key_functions = tuple(
            sorted(
                (scheme, _pool_key_function_identity(function))
                for scheme, function in poolmanager.key_fn_by_scheme.items()
            )
        )
        if tuple(scheme for scheme, _value in pool_classes) != ("http", "https"):
            raise ValueError("formal DashScope connection pool classes changed")
        if tuple(scheme for scheme, _value in key_functions) != ("http", "https"):
            raise ValueError("formal DashScope connection key policy changed")
        state_objects.extend(
            (
                adapter,
                adapter.config,
                adapter.proxy_manager,
                adapter.max_retries,
                poolmanager,
                poolmanager.headers,
                poolmanager.connection_pool_kw,
                poolmanager.pools,
                poolmanager.pool_classes_by_scheme,
                poolmanager.key_fn_by_scheme,
            )
        )
        adapter_payloads.append(
            {
                "connection_pool": dict(poolmanager.connection_pool_kw),
                "key_functions": [
                    [scheme, identity] for scheme, identity in key_functions
                ],
                "max_retries": repr(adapter.max_retries),
                "pool_classes": [list(item) for item in pool_classes],
                "route": route,
            }
        )
    payload = {
        "adapters": adapter_payloads,
        "headers": [list(item) for item in _DEFAULT_SESSION_HEADERS],
        "max_redirects": requests.models.DEFAULT_REDIRECT_LIMIT,
        "policy_version": "dashscope-default-session-v1",
        "requests_version": _REQUESTS_VERSION,
        "trust_env": False,
        "verify": True,
    }
    return sha256_bytes(canonical_json_bytes(payload)), tuple(state_objects)


class EmbeddingBackend(Protocol):
    """Embedding source used by deterministic index builders."""

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_images(self, paths: Sequence[Path]) -> np.ndarray: ...


class ImageBytesEmbeddingBackend(Protocol):
    """Optional capability which avoids reopening an already-audited path."""

    def embed_image_bytes(self, images: Sequence[bytes]) -> np.ndarray: ...


class DashScopeEmbeddingClient:
    """Minimal client for ``qwen3-vl-embedding`` independent embeddings."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        session: Any | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        backoff_base: float = 1.0,
        max_attempts: int = _MAX_ATTEMPTS,
        timeout: float = 120.0,
    ) -> None:
        self._api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        owns_session = session is None
        self._session = _REQUESTS_SESSION_TYPE() if owns_session else session
        if owns_session:
            # Formal traffic must not inherit mutable environment proxy settings.
            self._session.trust_env = False
        self._sleeper = sleeper
        self._backoff_base = backoff_base
        self._max_attempts = max_attempts
        self._timeout = timeout
        self._last_request_id: str | None = None
        self._last_request_ids: list[str] = []
        if owns_session:
            _FORMAL_SESSION_AUTHORITY.issue(
                self,
                self._session,
                issuer=_FORMAL_SESSION_ISSUER,
            )

    @property
    def model(self) -> str:
        """Pinned service-side model identity used by formal runtime checks."""
        return EMBEDDING_MODEL

    @property
    def execution_location(self) -> Literal["remote"]:
        """Image bytes leave the local process for the pinned API endpoint."""

        return "remote"

    @property
    def formal_runtime_binding_sha256(self) -> str:
        """Bind the reviewed HTTP client policy without exposing credentials."""

        session_policy_sha256 = _FORMAL_SESSION_AUTHORITY.validate(self, self._session)
        if self._sleeper is not time.sleep:
            raise ValueError("formal DashScope runtime requires the default sleeper")
        if (
            not isinstance(self._max_attempts, int)
            or self._max_attempts < 1
            or not isinstance(self._timeout, (int, float))
            or self._timeout <= 0
            or not isinstance(self._backoff_base, (int, float))
            or self._backoff_base < 0
        ):
            raise ValueError("formal DashScope retry policy is invalid")
        payload = {
            "backoff_base": float(self._backoff_base),
            "dimension": EMBEDDING_DIMENSION,
            "endpoint": _FORMAL_DASHSCOPE_ENDPOINT,
            "execution_location": self.execution_location,
            "image_batch_size": IMAGE_BATCH_SIZE,
            "max_attempts": self._max_attempts,
            "model": EMBEDDING_MODEL,
            "policy_version": "dashscope-embedding-client-v2",
            "requests_version": _REQUESTS_VERSION,
            "session_policy_sha256": session_policy_sha256,
            "text_batch_size": TEXT_BATCH_SIZE,
            "timeout_seconds": float(self._timeout),
        }
        return sha256_bytes(canonical_json_bytes(payload))

    @property
    def last_request_id(self) -> str | None:
        """Request id of the most recent successful HTTP batch."""
        return self._last_request_id

    @property
    def last_request_ids(self) -> tuple[str, ...]:
        """Request ids produced by the most recent public embedding call."""
        return tuple(self._last_request_ids)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed text inputs in DashScope's documented twenty-item batches."""
        return self._embed_batched(texts, TEXT_BATCH_SIZE, self._text_content)

    def embed_images(self, paths: Sequence[Path]) -> np.ndarray:
        """Embed image inputs in DashScope's documented five-image batches."""
        return self._embed_batched(paths, IMAGE_BATCH_SIZE, self._image_content)

    def embed_image_bytes(self, images: Sequence[bytes]) -> np.ndarray:
        """Embed immutable image snapshots without another filesystem lookup."""
        return self._embed_batched(images, IMAGE_BATCH_SIZE, self._image_bytes_content)

    def _embed_batched(
        self,
        values: Sequence[Any],
        batch_size: int,
        content_builder: Callable[[Any], dict[str, str]],
    ) -> np.ndarray:
        if not values:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

        self._last_request_ids = []
        batches: list[np.ndarray] = []
        for start in range(0, len(values), batch_size):
            contents = [
                content_builder(value) for value in values[start : start + batch_size]
            ]
            batches.append(self._request_embeddings(contents))
            if self._last_request_id is None:
                raise EmbeddingResponseError("DashScope response is missing request_id")
            self._last_request_ids.append(self._last_request_id)
        return np.vstack(batches).astype(np.float32, copy=False)

    @staticmethod
    def _text_content(text: str) -> dict[str, str]:
        _validate_text_input(text)
        return {"text": text}

    @staticmethod
    def _image_content(path: Path) -> dict[str, str]:
        image_path = Path(path)
        try:
            image_bytes = image_path.read_bytes()
        except FileNotFoundError as error:
            raise FileNotFoundError(f"image does not exist: {image_path}") from error
        if not image_bytes:
            raise ValueError(f"image is empty: {image_path}")
        if len(image_bytes) > _MAX_IMAGE_BYTES:
            raise ValueError(
                f"image must not exceed {_MAX_IMAGE_BYTES} bytes: {image_path}"
            )

        return DashScopeEmbeddingClient._image_bytes_content(image_bytes)

    @staticmethod
    def _image_bytes_content(image_bytes: bytes) -> dict[str, str]:
        if not isinstance(image_bytes, bytes):
            raise TypeError("image snapshots must be bytes")
        if not image_bytes:
            raise ValueError("image snapshot is empty")
        if len(image_bytes) > _MAX_IMAGE_BYTES:
            raise ValueError(f"image snapshot must not exceed {_MAX_IMAGE_BYTES} bytes")
        try:
            from io import BytesIO

            with Image.open(BytesIO(image_bytes)) as image:
                image_format = image.format
                image.verify()
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError("image snapshot cannot be decoded") from error
        media_type = _IMAGE_MEDIA_TYPES.get(image_format or "")
        if media_type is None:
            raise ValueError(f"unsupported image format: {image_format!r}")
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return {"image": f"data:{media_type};base64,{encoded}"}

    def _request_embeddings(self, contents: list[dict[str, str]]) -> np.ndarray:
        if not self._api_key:
            raise EmbeddingRequestError("DASHSCOPE_API_KEY is required for embeddings")
        request = {
            "model": EMBEDDING_MODEL,
            "input": {"contents": contents},
            "parameters": {"dimension": EMBEDDING_DIMENSION},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(self._max_attempts):
            _FORMAL_SESSION_AUTHORITY.validate_if_issued(self, self._session)
            try:
                response = self._session.post(
                    _FORMAL_DASHSCOPE_ENDPOINT,
                    headers=headers,
                    json=request,
                    timeout=self._timeout,
                )
            except (requests.RequestException, TimeoutError, ConnectionError) as error:
                _FORMAL_SESSION_AUTHORITY.validate_if_issued(self, self._session)
                if attempt == self._max_attempts - 1:
                    raise EmbeddingRequestError(
                        "DashScope request failed after retries"
                    ) from error
                self._sleeper(self._backoff_base * (2**attempt))
                continue

            _FORMAL_SESSION_AUTHORITY.validate_if_issued(self, self._session)

            if 200 <= response.status_code < 300:
                return self._parse_response(response, len(contents))
            if response.status_code != 429 and not 500 <= response.status_code < 600:
                raise EmbeddingRequestError(
                    f"DashScope rejected request with HTTP {response.status_code}"
                )
            if attempt == self._max_attempts - 1:
                raise EmbeddingRequestError(
                    f"DashScope unavailable after HTTP {response.status_code}"
                )
            self._sleeper(self._retry_delay(response, attempt))

        raise AssertionError("retry loop must either return or raise")

    def _parse_response(self, response: Any, expected_count: int) -> np.ndarray:
        try:
            payload = response.json()
        except (TypeError, ValueError) as error:
            raise EmbeddingResponseError(
                "DashScope response is not valid JSON"
            ) from error
        if not isinstance(payload, dict) or not payload.get("request_id"):
            raise EmbeddingResponseError("DashScope response is missing request_id")

        output = payload.get("output")
        if not isinstance(output, dict):
            raise EmbeddingResponseError("DashScope response output must be an object")
        embeddings = output.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != expected_count:
            raise EmbeddingResponseError(
                "DashScope response has an unexpected embedding count"
            )

        vectors: list[np.ndarray | None] = [None] * expected_count
        for item in embeddings:
            if not isinstance(item, dict):
                raise EmbeddingResponseError("DashScope embedding item is invalid")
            index = item.get("index")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < expected_count
            ):
                raise EmbeddingResponseError("DashScope embedding index is invalid")
            if vectors[index] is not None:
                raise EmbeddingResponseError(
                    "DashScope response contains duplicate embedding indices"
                )
            vectors[index] = _normalise_vector(
                item.get("embedding"), error_type=EmbeddingResponseError
            )

        if any(vector is None for vector in vectors):
            raise EmbeddingResponseError("DashScope response omits embedding indices")
        self._last_request_id = str(payload["request_id"])
        return np.vstack(vectors).astype(np.float32, copy=False)

    def _retry_delay(self, response: Any, attempt: int) -> float:
        retry_after = getattr(response, "headers", {}).get("Retry-After")
        try:
            delay = float(retry_after)
        except (TypeError, ValueError):
            delay = self._backoff_base * (2**attempt)
        return max(0.0, delay)


class EmbeddingCache:
    """A WAL cache keyed by model, dimension, modality, and raw input bytes."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    model TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    modality TEXT NOT NULL CHECK(modality IN ('text','image')),
                    input_sha256 TEXT NOT NULL,
                    input_bytes INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    request_id TEXT NOT NULL,
                    PRIMARY KEY(model, dimension, modality, input_sha256)
                )
                """
            )

    def get(self, modality: str, value: str | Path) -> np.ndarray | None:
        raw_input = _raw_input_bytes(modality, value)
        digest = hashlib.sha256(raw_input).hexdigest()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT input_bytes, vector FROM embeddings
                WHERE model = ? AND dimension = ? AND modality = ? AND input_sha256 = ?
                """,
                (EMBEDDING_MODEL, EMBEDDING_DIMENSION, modality, digest),
            ).fetchone()
        if row is None:
            return None
        input_size, vector_blob = row
        if input_size != len(raw_input):
            raise CacheCorruptionError("cached input size does not match input hash")
        return _normalise_vector_blob(vector_blob)

    def put_many(
        self,
        modality: str,
        entries: Iterable[tuple[str | Path, np.ndarray, str]],
    ) -> None:
        serialized_rows = []
        for value, vector, request_id in entries:
            raw_input = _raw_input_bytes(modality, value)
            normalized = _normalise_vector(vector, error_type=CacheCorruptionError)
            serialized_rows.append(
                (
                    EMBEDDING_MODEL,
                    EMBEDDING_DIMENSION,
                    modality,
                    hashlib.sha256(raw_input).hexdigest(),
                    len(raw_input),
                    normalized.astype(np.float32, copy=False).tobytes(),
                    request_id,
                )
            )
        if not serialized_rows:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO embeddings
                (model, dimension, modality, input_sha256, input_bytes, vector, request_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                serialized_rows,
            )

    def replace_model_dimension_with_text(
        self, value: str, vector: np.ndarray, request_id: str
    ) -> None:
        """Atomically replace all model cache rows with one fresh text baseline."""
        raw_input = _raw_input_bytes("text", value)
        normalized = _normalise_vector(vector, error_type=CacheCorruptionError)
        row = (
            EMBEDDING_MODEL,
            EMBEDDING_DIMENSION,
            "text",
            hashlib.sha256(raw_input).hexdigest(),
            len(raw_input),
            normalized.astype(np.float32, copy=False).tobytes(),
            request_id,
        )
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM embeddings WHERE model = ? AND dimension = ?",
                (EMBEDDING_MODEL, EMBEDDING_DIMENSION),
            )
            connection.execute(
                """
                INSERT INTO embeddings
                (model, dimension, modality, input_sha256, input_bytes, vector, request_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)


class OpenCLIPConfig(BaseModel):
    """Exact inference contract for a reviewed local 1024-d CLIP track."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1] = 1
    model_name: Literal["RN50", "xlm-roberta-large-ViT-H-14"] = "RN50"
    pretrained: Literal["openai", "frozen_laion5b_s13b_b90k"] = "openai"
    tokenizer_kind: Literal["simple_bpe", "hf_local"] = "simple_bpe"
    device: Literal["cpu"] = "cpu"
    precision: Literal["fp32"] = "fp32"
    output_dimension: Literal[1024] = 1024
    torch_version: str
    transformers_version: str | None = None
    image_mean: tuple[float, float, float] = (
        0.48145466,
        0.4578275,
        0.40821073,
    )
    image_std: tuple[float, float, float] = (
        0.26862954,
        0.26130258,
        0.27577711,
    )
    image_interpolation: Literal["bicubic"] = "bicubic"
    image_resize_mode: Literal["shortest"] = "shortest"

    @field_validator("image_mean", "image_std", mode="before")
    @classmethod
    def coerce_json_vectors(cls, value):
        return tuple(value) if isinstance(value, list) else value

    @field_validator("torch_version")
    @classmethod
    def validate_torch_version(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("torch_version must be nonblank and canonical")
        return value

    @field_validator("transformers_version")
    @classmethod
    def validate_transformers_version(cls, value: str | None) -> str | None:
        if value is not None and (not value or value != value.strip()):
            raise ValueError("transformers_version must be absent or canonical")
        return value

    @model_validator(mode="after")
    def validate_model_track(self):
        if self.model_name == "RN50":
            expected = ("openai", "simple_bpe", None)
        else:
            expected = (
                "frozen_laion5b_s13b_b90k",
                "hf_local",
                self.transformers_version,
            )
            if self.transformers_version is None:
                raise ValueError(
                    "multilingual OpenCLIP requires an exact transformers version"
                )
        if (
            self.pretrained,
            self.tokenizer_kind,
            self.transformers_version,
        ) != expected:
            raise ValueError("OpenCLIP model and tokenizer configuration is invalid")
        return self


class _OpenCLIPRuntimeStore:
    """Keep loaded state outside instances so callers cannot inject a model."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._values: WeakKeyDictionary[
            object, tuple[object, object, object, object]
        ] = WeakKeyDictionary()

    def get(self, owner: object) -> tuple[object, object, object, object] | None:
        with self._lock:
            return self._values.get(owner)

    def issue(
        self,
        owner: object,
        runtime: tuple[object, object, object, object],
    ) -> None:
        with self._lock:
            self._values[owner] = runtime


_OPENCLIP_RUNTIMES = _OpenCLIPRuntimeStore()


class OpenCLIPEmbeddingBackend:
    """Artifact-locked, fully local OpenCLIP image/text embeddings."""

    execution_location: Literal["local"] = "local"

    def __init__(self, artifact: VerifiedModelArtifact) -> None:
        if not artifact.external_sha256_verified:
            raise ValueError(
                "local embedding runtime requires an externally locked manifest"
            )
        if artifact.manifest.artifact_kind != "multimodal_embedding":
            raise ValueError(
                "OpenCLIP backend requires a multimodal embedding artifact"
            )
        if artifact.manifest.backend_name != "open-clip-torch":
            raise ValueError(
                "OpenCLIP backend requires an open-clip-torch model manifest"
            )
        self._artifact = artifact
        self._lock = RLock()
        self._config = self._load_config()
        self._validate_tokenizer_artifacts()
        expected_model = (
            f"open-clip:{self._config.model_name}:{self._config.pretrained}"
        )
        if artifact.manifest.model_id != expected_model:
            raise ValueError("OpenCLIP model_id does not match its locked config")

    @property
    def model(self) -> str:
        return self._artifact.manifest.model_id

    @property
    def _runtime(self) -> tuple[object, object, object, object] | None:
        if "_runtime" in vars(self):
            raise ValueError("OpenCLIP runtime state was injected outside the loader")
        return _OPENCLIP_RUNTIMES.get(self)

    @property
    def formal_runtime_binding_sha256(self) -> str:
        self._artifact.verify_files()
        self._load_runtime()
        identity = {
            "artifact_runtime": self._artifact.runtime_binding.model_dump(mode="json"),
            "config": self._config.model_dump(mode="json"),
            "execution_location": self.execution_location,
            "policy_version": "open-clip-local-backend-v1",
        }
        return sha256_bytes(canonical_json_bytes(identity))

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        for value in texts:
            _validate_text_input(value)
        if not texts:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
        model, _preprocess, tokenizer, torch = self._load_runtime()
        with torch.inference_mode():
            tokens = tokenizer(list(texts))
            encoded = model.encode_text(tokens)
        return self._to_numpy(encoded, len(texts))

    def embed_images(self, paths: Sequence[Path]) -> np.ndarray:
        snapshots: list[bytes] = []
        for raw_path in paths:
            path = Path(raw_path)
            try:
                content = path.read_bytes()
            except OSError as error:
                raise ValueError(
                    f"local embedding image cannot be read: {path}"
                ) from error
            snapshots.append(content)
        return self.embed_image_bytes(snapshots)

    def embed_image_bytes(self, images: Sequence[bytes]) -> np.ndarray:
        if not images:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
        model, preprocess, _tokenizer, torch = self._load_runtime()
        tensors = [preprocess(_decode_local_image(content)) for content in images]
        with torch.inference_mode():
            batch = torch.stack(tensors)
            encoded = model.encode_image(batch)
        return self._to_numpy(encoded, len(images))

    def _load_config(self) -> OpenCLIPConfig:
        content = self._artifact.read_artifact_bytes("config", max_bytes=64 * 1024)
        try:
            raw = json.loads(content.decode("utf-8"))
            config = OpenCLIPConfig.model_validate(raw, strict=True)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise ValueError("OpenCLIP config is invalid") from error
        if content != canonical_json_bytes(config.model_dump(mode="json")):
            raise ValueError("OpenCLIP config must be canonical JSON")
        return config

    def _load_runtime(self) -> tuple[object, object, object, object]:
        loaded = self._runtime
        if loaded is not None:
            return loaded
        with self._lock:
            loaded = self._runtime
            if loaded is not None:
                return loaded
            try:
                import open_clip
                import torch
            except ImportError as error:
                raise LocalEmbeddingRuntimeUnavailableError(
                    "open_clip_torch and torch are required for local embeddings"
                ) from error
            try:
                open_clip_version = metadata.version("open_clip_torch")
                torch_version = metadata.version("torch")
            except metadata.PackageNotFoundError as error:
                raise LocalEmbeddingRuntimeUnavailableError(
                    "local embedding package metadata is unavailable"
                ) from error
            if open_clip_version != self._artifact.manifest.backend_version:
                raise ValueError(
                    "installed open_clip_torch version does not match the manifest"
                )
            if torch_version != self._config.torch_version:
                raise ValueError("installed torch version does not match the config")
            if self._config.transformers_version is not None:
                try:
                    transformers_version = metadata.version("transformers")
                except metadata.PackageNotFoundError as error:
                    raise LocalEmbeddingRuntimeUnavailableError(
                        "transformers is required for multilingual local embeddings"
                    ) from error
                if transformers_version != self._config.transformers_version:
                    raise ValueError(
                        "installed transformers version does not match the config"
                    )
            self._artifact.verify_files()
            model_kwargs: dict[str, object] = {}
            if self._config.tokenizer_kind == "hf_local":
                model_config = open_clip.get_model_config(self._config.model_name)
                if not isinstance(model_config, dict):
                    raise ValueError("OpenCLIP model config is unavailable")
                text_config = dict(model_config.get("text_cfg", {}))
                text_config["hf_model_name"] = str(self._tokenizer_directory())
                text_config["hf_tokenizer_name"] = str(self._tokenizer_directory())
                text_config["hf_model_pretrained"] = False
                model_kwargs["text_cfg"] = text_config
                model_kwargs["pretrained_text"] = False
            model, _unused_train, preprocess = open_clip.create_model_and_transforms(
                self._config.model_name,
                pretrained=str(self._artifact.artifact_path("weights")),
                device=self._config.device,
                precision=self._config.precision,
                image_mean=self._config.image_mean,
                image_std=self._config.image_std,
                image_interpolation=self._config.image_interpolation,
                image_resize_mode=self._config.image_resize_mode,
                **model_kwargs,
            )
            if self._config.tokenizer_kind == "simple_bpe":
                tokenizer = open_clip.SimpleTokenizer(
                    bpe_path=str(self._artifact.artifact_path("tokenizer"))
                )
            else:
                try:
                    from open_clip.tokenizer import HFTokenizer
                except ImportError as error:
                    raise LocalEmbeddingRuntimeUnavailableError(
                        "OpenCLIP HF tokenizer runtime is unavailable"
                    ) from error
                tokenizer = HFTokenizer(
                    str(self._tokenizer_directory()),
                    local_files_only=True,
                )
            model.eval()
            self._artifact.verify_files()
            loaded = (model, preprocess, tokenizer, torch)
            _OPENCLIP_RUNTIMES.issue(self, loaded)
            return loaded

    def _validate_tokenizer_artifacts(self) -> None:
        if self._config.tokenizer_kind == "simple_bpe":
            self._artifact.descriptor("tokenizer")
            return
        required = {
            "tokenizer",
            "tokenizer_config",
            "tokenizer_model",
            "transformer_config",
        }
        roles = {item.role for item in self._artifact.manifest.artifacts}
        if not required.issubset(roles):
            raise ValueError(
                "multilingual OpenCLIP requires tokenizer, tokenizer_config, "
                "and tokenizer_model artifacts"
            )
        tokenizer_descriptors = tuple(
            item
            for item in self._artifact.manifest.artifacts
            if item.role == "transformer_config"
            or item.role == "tokenizer"
            or item.role.startswith("tokenizer_")
            or item.role == "special_tokens_map"
        )
        parents = {
            self._artifact.artifact_path(item.role).parent
            for item in tokenizer_descriptors
        }
        if len(parents) != 1:
            raise ValueError(
                "multilingual tokenizer artifacts must share one directory"
            )
        directory = next(iter(parents))
        registered = {
            self._artifact.artifact_path(item.role).name
            for item in tokenizer_descriptors
        }
        entries = tuple(directory.iterdir())
        if any(path.is_symlink() or not path.is_file() for path in entries):
            raise ValueError(
                "multilingual tokenizer directory must contain regular files only"
            )
        observed = {path.name for path in entries}
        if observed != registered:
            raise ValueError(
                "multilingual tokenizer directory contains unregistered files"
            )

    def _tokenizer_directory(self) -> Path:
        return self._artifact.artifact_path("tokenizer").parent

    @staticmethod
    def _to_numpy(value: object, expected_rows: int) -> np.ndarray:
        try:
            array = value.detach().cpu().numpy().astype(np.float32, copy=False)
        except (AttributeError, TypeError, ValueError) as error:
            raise EmbeddingResponseError(
                "OpenCLIP returned an invalid tensor"
            ) from error
        return _normalise_matrix(array, expected_rows)


def _decode_local_image(content: bytes) -> Image.Image:
    if not isinstance(content, bytes):
        raise TypeError("image snapshots must be bytes")
    if not content:
        raise ValueError("image snapshot is empty")
    if len(content) > _MAX_IMAGE_BYTES:
        raise ValueError(f"image snapshot must not exceed {_MAX_IMAGE_BYTES} bytes")
    try:
        with Image.open(BytesIO(content)) as image:
            image.load()
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("image snapshot cannot be decoded") from error


class FormalEmbeddingBackend:
    """Uncached embedding runtime with a pinned, freshly checked canary.

    Formal retrieval deliberately has no cache object and therefore cannot
    consume a syntactically valid vector inserted into a writable SQLite file.
    """

    cache_policy = "bypass"
    runtime_policy_version = "formal-embedding-runtime-v1"

    def __init__(
        self,
        backend: EmbeddingBackend,
        *,
        canary_text: str,
        canary_vector: np.ndarray,
    ) -> None:
        _validate_text_input(canary_text)
        self._backend = backend
        self._canary_text = canary_text
        self._canary_vector = _normalise_vector(canary_vector, error_type=ValueError)

    @property
    def model(self) -> str:
        model = getattr(self._backend, "model", None)
        if not isinstance(model, str) or not model.strip():
            raise ValueError("formal embedding backend model identity is invalid")
        return model

    @property
    def execution_location(self) -> Literal["local", "remote"]:
        return embedding_execution_location(self._backend)

    @property
    def canary_text(self) -> str:
        return self._canary_text

    @property
    def canary_vector(self) -> np.ndarray:
        return self._canary_vector.copy()

    @property
    def runtime_binding_sha256(self) -> str:
        upstream_binding = getattr(self._backend, "formal_runtime_binding_sha256", None)
        if (
            not isinstance(upstream_binding, str)
            or len(upstream_binding) != 64
            or any(
                character not in "0123456789abcdef" for character in upstream_binding
            )
        ):
            raise ValueError(
                "formal embedding runtime requires a bound upstream backend"
            )
        identity = {
            "cache_policy": self.cache_policy,
            "canary_text_sha256": sha256_bytes(self._canary_text.encode("utf-8")),
            "canary_vector_sha256": sha256_bytes(
                np.ascontiguousarray(self._canary_vector, dtype=np.float32).tobytes()
            ),
            "dimension": EMBEDDING_DIMENSION,
            "execution_location": self.execution_location,
            "model": self.model,
            "policy_version": self.runtime_policy_version,
            "upstream_runtime_sha256": upstream_binding,
        }
        return sha256_bytes(canonical_json_bytes(identity))

    @property
    def formal_ready(self) -> bool:
        try:
            return _FORMAL_CANARY_RECEIPTS.get(self) == self.runtime_binding_sha256
        except Exception:
            return False

    def verify_canary(self, *, force: bool = False) -> None:
        # A receipt is issued only by a successful live check.  In particular,
        # assigning a similarly named instance attribute cannot promote a
        # backend into the formal path.
        _FORMAL_CANARY_RECEIPTS.pop(self, None)
        current = _normalise_matrix(self._backend.embed_texts([self._canary_text]), 1)[
            0
        ]
        cosine = float(np.dot(current, self._canary_vector))
        if cosine < _CANARY_MIN_COSINE:
            raise ModelDriftError(
                f"embedding canary cosine {cosine:.8f} is below {_CANARY_MIN_COSINE}"
            )
        _FORMAL_CANARY_RECEIPTS[self] = self.runtime_binding_sha256

    def _require_verified(self) -> None:
        if not self.formal_ready:
            raise ModelDriftError("formal embedding canary has not been verified")

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        self._require_verified()
        for text in texts:
            _validate_text_input(text)
        return _normalise_matrix(self._backend.embed_texts(texts), len(texts))

    def embed_images(self, paths: Sequence[Path]) -> np.ndarray:
        self._require_verified()
        return _normalise_matrix(self._backend.embed_images(paths), len(paths))

    def embed_image_bytes(self, images: Sequence[bytes]) -> np.ndarray:
        self._require_verified()
        embed = getattr(self._backend, "embed_image_bytes", None)
        if not callable(embed):
            raise EmbeddingRequestError(
                "formal image runtime does not support immutable byte inputs"
            )
        return _normalise_matrix(embed(images), len(images))


class CachedEmbeddingBackend:
    """Cache-first embedding backend with an uncached model-drift canary."""

    def __init__(
        self,
        backend: EmbeddingBackend,
        cache: EmbeddingCache,
        *,
        canary_text: str | None = None,
        canary_vector: np.ndarray | None = None,
    ) -> None:
        if (canary_text is None) != (canary_vector is None):
            raise ValueError("canary_text and canary_vector must be supplied together")
        if canary_text is not None:
            _validate_text_input(canary_text)
        self._backend = backend
        self._cache = cache
        self._canary_text = canary_text
        self._canary_vector = (
            _normalise_vector(canary_vector, error_type=ValueError)
            if canary_vector is not None
            else None
        )
        self._canary_verified = False
        self._cache_hits_by_modality = {"image": 0, "text": 0}
        self._api_calls_by_modality = {"image": 0, "text": 0}

    @property
    def model(self) -> str:
        """Model identity inherited from the wrapped backend."""
        model = getattr(self._backend, "model", EMBEDDING_MODEL)
        if not isinstance(model, str) or not model.strip():
            raise ValueError("embedding backend model identity is invalid")
        return model

    @property
    def execution_location(self) -> Literal["local", "remote"]:
        return embedding_execution_location(self._backend)

    @property
    def cache_hits(self) -> int:
        """Number of input vectors resolved from cache by this backend instance."""
        return sum(self._cache_hits_by_modality.values())

    @property
    def api_calls(self) -> int:
        """Number of actual underlying embedding requests made by this instance."""
        return sum(self._api_calls_by_modality.values())

    @property
    def cache_hits_by_modality(self) -> dict[str, int]:
        """Cache-hit input-vector counts split by image and text modality."""
        return dict(self._cache_hits_by_modality)

    @property
    def api_calls_by_modality(self) -> dict[str, int]:
        """Underlying embedding request counts split by image and text modality."""
        return dict(self._api_calls_by_modality)

    @property
    def canary_text(self) -> str | None:
        """Configured canary text, if this backend can perform drift checks."""
        return self._canary_text

    @property
    def canary_vector(self) -> np.ndarray | None:
        """A defensive copy of the configured canary vector, if present."""
        return None if self._canary_vector is None else self._canary_vector.copy()

    def establish_canary(self, text: str) -> np.ndarray:
        """Bypass cache to establish a fresh model-drift baseline for ``text``."""
        _validate_text_input(text)
        self._api_calls_by_modality["text"] += 1
        vector = _normalise_matrix(self._backend.embed_texts([text]), 1)[0]
        self._cache.replace_model_dimension_with_text(
            text, vector, _backend_request_id(self._backend)
        )
        self._canary_text = text
        self._canary_vector = vector
        self._canary_verified = True
        return vector.reshape(1, -1).copy()

    def verify_canary(self, *, force: bool = False) -> None:
        """Perform the uncached drift check required before a formal run."""

        if self._canary_text is None or self._canary_vector is None:
            raise ValueError("formal embedding runtime requires a configured canary")
        if force:
            self._canary_verified = False
        self._verify_canary()

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        for text in texts:
            _validate_text_input(text)
        return self._embed("text", texts, self._backend.embed_texts)

    def embed_images(self, paths: Sequence[Path]) -> np.ndarray:
        normalized_paths = [Path(path) for path in paths]
        return self._embed("image", normalized_paths, self._backend.embed_images)

    def _embed(
        self,
        modality: str,
        values: Sequence[str] | Sequence[Path],
        embed_missing: Callable[[Sequence[Any]], np.ndarray],
    ) -> np.ndarray:
        if not values:
            return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

        resolved: list[np.ndarray | None] = []
        missing_positions: dict[str, list[int]] = {}
        missing_values: dict[str, str | Path] = {}
        for position, value in enumerate(values):
            cached = self._cache.get(modality, value)
            resolved.append(cached)
            if cached is None:
                digest = _input_digest(modality, value)
                missing_positions.setdefault(digest, []).append(position)
                missing_values.setdefault(digest, value)
            else:
                self._cache_hits_by_modality[modality] += 1
        if not missing_values:
            return np.vstack(resolved).astype(np.float32, copy=False)

        self._verify_canary()
        unique_values = list(missing_values.values())
        batch_size = TEXT_BATCH_SIZE if modality == "text" else IMAGE_BATCH_SIZE
        for start in range(0, len(unique_values), batch_size):
            batch_values = unique_values[start : start + batch_size]
            self._api_calls_by_modality[modality] += 1
            generated = _normalise_matrix(
                embed_missing(batch_values), len(batch_values)
            )
            request_id = _backend_request_id(self._backend)
            self._cache.put_many(
                modality,
                (
                    (value, vector, request_id)
                    for value, vector in zip(batch_values, generated, strict=True)
                ),
            )
            for value, vector in zip(batch_values, generated, strict=True):
                digest = _input_digest(modality, value)
                for position in missing_positions[digest]:
                    resolved[position] = vector
        return np.vstack(resolved).astype(np.float32, copy=False)

    def _verify_canary(self) -> None:
        if (
            self._canary_verified
            or self._canary_text is None
            or self._canary_vector is None
        ):
            return
        self._api_calls_by_modality["text"] += 1
        current = _normalise_matrix(self._backend.embed_texts([self._canary_text]), 1)[
            0
        ]
        cosine = float(np.dot(current, self._canary_vector))
        if cosine < _CANARY_MIN_COSINE:
            raise ModelDriftError(
                f"embedding canary cosine {cosine:.8f} is below {_CANARY_MIN_COSINE}"
            )
        self._canary_verified = True


def _raw_input_bytes(modality: str, value: str | Path) -> bytes:
    if modality == "text":
        if not isinstance(value, str):
            raise TypeError("text cache values must be strings")
        return value.encode("utf-8")
    if modality == "image":
        return Path(value).read_bytes()
    raise ValueError(f"unsupported embedding modality: {modality}")


def _validate_text_input(text: str) -> None:
    if not isinstance(text, str):
        raise TypeError("text embeddings require strings")
    if not text.strip():
        raise ValueError("text embedding input must not be blank")


def embedding_execution_location(
    backend: object,
) -> Literal["local", "remote"]:
    """Return an explicit data-residency contract; missing claims fail closed."""

    location = getattr(backend, "execution_location", None)
    if location not in {"local", "remote"}:
        raise ValueError(
            "embedding backend must declare execution_location as local or remote"
        )
    return location


def formal_embedding_runtime_binding(backend: object) -> str | None:
    """Return a formal build attestation for reviewed remote or local runtimes.

    Arbitrary protocol implementations remain useful for deterministic tests and
    diagnostic indexes, but their self-declared execution location is not a
    data-residency trust root.
    """

    if type(backend) is not FormalEmbeddingBackend:
        return None
    if type(getattr(backend, "_backend", None)) not in {
        DashScopeEmbeddingClient,
        OpenCLIPEmbeddingBackend,
    }:
        return None
    if not backend.formal_ready:
        return None
    try:
        binding = backend.runtime_binding_sha256
    except Exception:
        return None
    return binding


def _input_digest(modality: str, value: str | Path) -> str:
    return hashlib.sha256(_raw_input_bytes(modality, value)).hexdigest()


def _backend_request_id(backend: Any) -> str:
    request_id = getattr(backend, "last_request_id", None)
    if isinstance(request_id, str) and request_id:
        return request_id
    request_id = getattr(backend, "request_id", None)
    if isinstance(request_id, str) and request_id:
        return request_id
    return "backend"


def _normalise_matrix(value: Any, expected_rows: int) -> np.ndarray:
    try:
        vectors = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise EmbeddingResponseError(
            "embedding matrix is not numeric or rectangular"
        ) from error
    if vectors.shape != (expected_rows, EMBEDDING_DIMENSION):
        raise EmbeddingResponseError(
            f"embedding matrix must have shape ({expected_rows}, {EMBEDDING_DIMENSION})"
        )
    return np.vstack(
        [
            _normalise_vector(vector, error_type=EmbeddingResponseError)
            for vector in vectors
        ]
    ).astype(np.float32, copy=False)


def _normalise_vector(value: Any, *, error_type: type[Exception]) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise error_type("embedding vector is not numeric") from error
    if vector.shape != (EMBEDDING_DIMENSION,):
        raise error_type(f"embedding vector must have {EMBEDDING_DIMENSION} dimensions")
    if not np.all(np.isfinite(vector)):
        raise error_type("embedding vector contains non-finite values")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm == 0.0:
        raise error_type("embedding vector must have non-zero finite norm")
    return (vector / norm).astype(np.float32, copy=False)


def _normalise_vector_blob(value: Any) -> np.ndarray:
    if (
        not isinstance(value, bytes)
        or len(value) != EMBEDDING_DIMENSION * np.dtype(np.float32).itemsize
    ):
        raise CacheCorruptionError("cached vector blob has an invalid length")
    vector = np.frombuffer(value, dtype=np.float32).copy()
    normalized = _normalise_vector(vector, error_type=CacheCorruptionError)
    if not np.allclose(vector, normalized, rtol=1e-5, atol=1e-5):
        raise CacheCorruptionError("cached vector is not normalized")
    return normalized


__all__ = [
    "CachedEmbeddingBackend",
    "CacheCorruptionError",
    "DashScopeEmbeddingClient",
    "EmbeddingBackend",
    "EmbeddingError",
    "EmbeddingRequestError",
    "EmbeddingResponseError",
    "EmbeddingCache",
    "embedding_execution_location",
    "formal_embedding_runtime_binding",
    "FormalEmbeddingBackend",
    "ImageBytesEmbeddingBackend",
    "LocalEmbeddingRuntimeUnavailableError",
    "ModelDriftError",
    "OpenCLIPConfig",
    "OpenCLIPEmbeddingBackend",
]
