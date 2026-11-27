"""Search settings must not persist an embedding model the admin does not
control.

A tenant admin (and a self-hosted admin) can call
``POST /api/search-settings/set-new-search-settings`` with ``provider_type``
null and an arbitrary ``model_name``. Indexing and query embedding later send
that name to the shared model server. These tests fail while that name is
stored. They pass once the endpoint rejects it before creating a search-settings
row or touching document indices, in both multi-tenant and single-tenant mode.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from onyx.context.search.models import SearchSettingsCreationRequest
from onyx.db.enums import EmbeddingPrecision
from onyx.error_handling.exceptions import OnyxError
from onyx.server.manage.search_settings import set_new_search_settings

_ATTACKER_MODEL = "attacker-org/evil-embed"


def _creation_request() -> SearchSettingsCreationRequest:
    return SearchSettingsCreationRequest(
        model_name=_ATTACKER_MODEL,
        model_dim=768,
        normalize=True,
        query_prefix=None,
        passage_prefix=None,
        provider_type=None,
        index_name=None,
        multipass_indexing=False,
        embedding_precision=EmbeddingPrecision.FLOAT,
        enable_contextual_rag=False,
    )


def _assert_rejected_before_save(multi_tenant: bool) -> None:
    current = SimpleNamespace(
        model_name="nomic-ai/nomic-embed-text-v1",
        index_name="danswer_chunk_nomic_ai_nomic_embed_text_v1",
        id=1,
        final_embedding_dim=768,
        embedding_precision=EmbeddingPrecision.FLOAT,
    )
    with (
        patch(
            "onyx.server.manage.search_settings.MULTI_TENANT",
            multi_tenant,
        ),
        patch(
            "onyx.server.manage.search_settings.get_current_search_settings",
            return_value=current,
        ),
        patch(
            "onyx.server.manage.search_settings.get_secondary_search_settings",
            return_value=None,
        ),
        patch(
            "onyx.server.manage.search_settings.create_search_settings",
        ) as create_search_settings,
        patch(
            "onyx.server.manage.search_settings.get_all_document_indices",
        ) as get_all_document_indices,
        pytest.raises((OnyxError, HTTPException)),
    ):
        set_new_search_settings(
            _creation_request(),
            MagicMock(),
            MagicMock(),
        )

    assert create_search_settings.call_count == 0
    assert get_all_document_indices.call_count == 0


def test_set_new_search_settings_rejects_untrusted_model_in_multi_tenant() -> None:
    _assert_rejected_before_save(multi_tenant=True)


def test_set_new_search_settings_rejects_untrusted_model_in_single_tenant() -> None:
    _assert_rejected_before_save(multi_tenant=False)
