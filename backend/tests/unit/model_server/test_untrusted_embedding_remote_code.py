"""Untrusted embedding repos must not run inside the model server.

Search settings supply ``model_name`` to ``EmbeddingModel``. For a local model
(``provider_type`` is null) ``encode`` posts that name to
``/encoder/bi-encoder-embed``. ``get_embedding_model`` then constructs
``SentenceTransformer(..., trust_remote_code=True)``, and transformers 4.53.0
executes ``auto_map`` modeling code from the named repository.

The repository here is a hub id whose files are already in the huggingface
cache. ``huggingface_hub`` 0.35.3 reads that cache when the hub HEAD request
fails closed (``HF_HUB_OFFLINE``), which is the same snapshot path it uses
after a download. The configuration module records that it was executed.

These tests fail while that code runs. They pass once indexing, query
embedding, and the model-server route reject the repository before the module
is imported.
"""

import asyncio
import importlib
import json
import os
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import requests
from fastapi import HTTPException
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

# Env is read by huggingface_hub.constants at import. Set it before those
# imports, then the fixture also patches the constants in case another test
# imported them first.
_CACHE_DIR = Path("/tmp/onyx-untrusted-embed-cache")
os.environ["HF_HOME"] = str(_CACHE_DIR / "home")
os.environ["HF_HUB_CACHE"] = str(_CACHE_DIR / "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(_CACHE_DIR / "hub")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from model_server.encoders import _GLOBAL_MODELS_DICT  # noqa: E402
from model_server.encoders import route_bi_encoder_embed  # noqa: E402
from onyx.indexing.embedder import DefaultIndexingEmbedder  # noqa: E402
from onyx.natural_language_processing.search_nlp_models import (  # noqa: E402
    EmbeddingModel,
)
from shared_configs.enums import EmbedTextType  # noqa: E402
from shared_configs.model_server_models import EmbedRequest  # noqa: E402

_ATTACKER_MODEL = "attacker-org/evil-embed"
_COMMIT = "a" * 40
_MARKER_ENV = "ONYX_EMBED_RCE_MARKER"


def _hub_cache() -> Path:
    return Path(os.environ["HF_HUB_CACHE"])


def _apply_hub_constants() -> None:
    import huggingface_hub.constants as hf_constants
    from huggingface_hub.utils import reset_sessions

    cache = str(_hub_cache())
    hf_constants.HF_HUB_OFFLINE = True
    hf_constants.HF_HUB_CACHE = cache
    hf_constants.HUGGINGFACE_HUB_CACHE = cache
    reset_sessions()


def _clear_loaded_remote_module() -> None:
    """Drop transformers' copy of the attacker module.

    ``get_cached_module_file`` in transformers 4.53.0 copies a remote module
    once per commit and will not refresh it. A later test in the same process
    would otherwise reuse that import and skip the payload.
    """
    from transformers.dynamic_module_utils import HF_MODULES_CACHE

    cached_repo = Path(HF_MODULES_CACHE) / "transformers_modules" / "attacker-org"
    if cached_repo.exists():
        shutil.rmtree(cached_repo)
    for name in list(sys.modules):
        if "transformers_modules" in name:
            del sys.modules[name]
    importlib.invalidate_caches()


def _seed_attacker_repo() -> None:
    """Lay out the hub snapshot huggingface_hub 0.35.3 resolves offline.

    ``hf_hub_download`` reads ``refs/<revision>`` and then
    ``snapshots/<commit>/<filename>`` when the metadata request fails.
    """
    _apply_hub_constants()
    repo = _hub_cache() / "models--attacker-org--evil-embed"
    snapshot = repo / "snapshots" / _COMMIT
    snapshot.mkdir(parents=True, exist_ok=True)
    ref = repo / "refs" / "main"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(_COMMIT)

    (snapshot / "config.json").write_text(
        json.dumps(
            {
                "model_type": "evil_remote_code",
                "auto_map": {"AutoConfig": "configuration.EvilConfig"},
            }
        )
    )
    # Imported by transformers.dynamic_module_utils.get_class_in_module via
    # importlib.util.module_from_spec(...).loader.exec_module.
    # transformers 4.53.0 configuration_auto.py loads this module with
    # get_class_from_dynamic_module, then calls register_for_auto_class on the
    # returned class. Writing the marker in both places records execution on
    # the first import and on every later load of the cached module.
    (snapshot / "configuration.py").write_text(
        "import os\n"
        "\n"
        "def _record(stage):\n"
        "    marker = os.environ.get(" + repr(_MARKER_ENV) + ")\n"
        "    if marker:\n"
        "        with open(marker, 'a', encoding='utf-8') as handle:\n"
        "            handle.write(stage + '\\n')\n"
        "\n"
        "_record('import')\n"
        "\n"
        "class EvilConfig:\n"
        "    @classmethod\n"
        "    def register_for_auto_class(cls, *args, **kwargs):\n"
        "        _record('register_for_auto_class')\n"
        "\n"
        "    @classmethod\n"
        "    def from_pretrained(cls, *args, **kwargs):\n"
        "        _record('from_pretrained')\n"
        "        raise RuntimeError('custom config loaded')\n"
    )
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "secret": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(snapshot / "tokenizer.json"))


def _embed_request(text: str, text_type: EmbedTextType) -> EmbedRequest:
    return EmbedRequest(
        model_name=_ATTACKER_MODEL,
        texts=[text],
        max_context_length=512,
        normalize_embeddings=True,
        api_key=None,
        provider_type=None,
        text_type=text_type,
        manual_query_prefix=None,
        manual_passage_prefix=None,
        api_url=None,
        api_version=None,
        deployment_name=None,
        reduced_dimension=None,
    )


def _search_settings() -> SimpleNamespace:
    return SimpleNamespace(
        model_name=_ATTACKER_MODEL,
        normalize=True,
        query_prefix=None,
        passage_prefix=None,
        provider_type=None,
        api_key=None,
        api_url=None,
        api_version=None,
        deployment_name=None,
        reduced_dimension=None,
    )


@pytest.fixture
def attacker_repo(tmp_path: Path) -> Iterator[Path]:
    _GLOBAL_MODELS_DICT.clear()
    _clear_loaded_remote_module()
    marker = tmp_path / "executed"
    os.environ[_MARKER_ENV] = str(marker)
    _seed_attacker_repo()
    yield marker
    _GLOBAL_MODELS_DICT.clear()
    os.environ.pop(_MARKER_ENV, None)


def _assert_not_executed(marker: Path) -> None:
    assert not marker.exists(), (
        "untrusted embedding repository executed code in the model server"
    )


def _assert_embedding_rejected(marker: Path, exc: BaseException) -> None:
    _assert_not_executed(marker)
    # encode() turns a model-server HTTP error into httpx.HTTPError. Any other
    # exception means the saved model name never reached that check.
    assert type(exc).__name__ in {"HTTPError", "HTTPException", "OnyxError", "ValueError"}, (
        f"embedding failed before the model server could reject the repository: "
        f"{type(exc).__name__}: {exc}"
    )


def test_model_server_route_does_not_execute_untrusted_repo(
    attacker_repo: Path,
) -> None:
    request = MagicMock()
    request.app.state.gpu_type = "UNKNOWN"

    with pytest.raises(Exception):
        asyncio.run(
            route_bi_encoder_embed(
                request,
                _embed_request("tenant document text", EmbedTextType.PASSAGE),
            )
        )

    _assert_not_executed(attacker_repo)


class _ForwardedResponse:
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError(f"{self.status_code} error")
            error.response = self  # type: ignore[attr-defined]
            raise error


def _forward_post(url: str, headers: dict[str, str] | None = None, json: Any = None, **_: Any) -> _ForwardedResponse:
    assert url.endswith("/encoder/bi-encoder-embed")
    assert isinstance(json, dict)
    assert json["model_name"] == _ATTACKER_MODEL
    try:
        asyncio.run(
            route_bi_encoder_embed(
                MagicMock(app=MagicMock(state=MagicMock(gpu_type="UNKNOWN"))),
                EmbedRequest(**json),
            )
        )
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return _ForwardedResponse(exc.status_code, {"detail": detail})
    return _ForwardedResponse(200, {"embeddings": []})


@pytest.mark.parametrize(
    "text_type",
    [EmbedTextType.PASSAGE, EmbedTextType.QUERY],
)
def test_saved_model_name_does_not_execute_when_embedded(
    attacker_repo: Path,
    text_type: EmbedTextType,
) -> None:
    settings = _search_settings()
    if text_type == EmbedTextType.PASSAGE:
        model = DefaultIndexingEmbedder.from_db_search_settings(
            search_settings=settings  # type: ignore[arg-type]
        ).embedding_model
    else:
        model = EmbeddingModel.from_db_model(
            search_settings=settings,  # type: ignore[arg-type]
            server_host="http://model-server",
            server_port=9000,
        )

    assert model.model_name == _ATTACKER_MODEL
    assert model.provider_type is None

    with patch(
        "onyx.natural_language_processing.search_nlp_models.requests.post",
        _forward_post,
    ):
        try:
            model.encode(["other tenant document text"], text_type=text_type)
        except Exception as exc:
            _assert_embedding_rejected(attacker_repo, exc)
        else:
            _assert_not_executed(attacker_repo)
            pytest.fail("embedding an untrusted model returned vectors")
