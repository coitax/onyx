"""Direct Slack thread fetches must use the same channel checks as keyword search.

These tests fail while a pasted Slack permalink can be read with the search
token before private-channel, DM, group-DM, and excluded-channel checks run.
They pass once those checks gate ``conversations.replies`` for the channel in
the URL.

The live Slack bot path is ``SearchTool._run_slack_search``, which always calls
``slack_retrieval`` with ``team_id=None``. ``SlackFederatedConnector.search``
is the other caller: it looks up a team id, then calls the same retrieval.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from slack_sdk.web.slack_response import SlackResponse

from onyx.context.search.federated.slack_search import query_slack
from onyx.context.search.models import ChunkIndexRequest
from onyx.context.search.models import IndexFilters
from onyx.context.search.models import InferenceChunk
from onyx.context.search.models import PersonaSearchInfo
from onyx.federated_connectors.slack.federated_connector import (
    SlackFederatedConnector,
)
from onyx.indexing.models import MultipassConfig
from onyx.onyxbot.slack.models import ChannelType
from onyx.onyxbot.slack.models import SlackContext
from onyx.tools.tool_implementations.search.search_tool import SearchTool

# Obvious placeholders. These are not credentials.
ACCESS_TOKEN = "placeholder-slack-access-token"
BOT_TOKEN = "placeholder-slack-bot-token"
CONNECTOR_CLIENT_ID = "placeholder-slack-client-id"
CONNECTOR_CLIENT_SECRET = "placeholder-slack-client-secret"

RAW_TS = "1761955200000000"
THREAD_TS = "1761955200.000000"
REPLY_TS = "1761955201.000000"

PUBLIC_CHANNEL = "C0PUBLICMSG"
PRIVATE_CHANNEL = "C0PRIVATETH"
OTHER_PRIVATE_CHANNEL = "C0OTHERPRIV"
DM_CHANNEL = "D0DIRECTMSG"
MPIM_CHANNEL = "G0GROUPDM01"
EXCLUDED_CHANNEL = "C0SECUREEXC"
ASKING_PUBLIC_CHANNEL = "C0PUBLICASK"

PUBLIC_SECRET = "public-thread-body-epsilon"
PRIVATE_SECRET = "private-thread-body-alpha"
PRIVATE_REPLY = "private-thread-reply-alpha"
OTHER_PRIVATE_SECRET = "other-private-thread-body-zeta"
DM_SECRET = "dm-thread-body-beta"
MPIM_SECRET = "group-dm-thread-body-gamma"
EXCLUDED_SECRET = "excluded-channel-body-delta"

CONTENT_METHODS = frozenset({"conversations_replies", "conversations_history"})


@dataclass
class ApiCall:
    method: str
    token: str
    channel: str | None = None
    ts: str | None = None


@dataclass
class ApiState:
    calls: list[ApiCall] = field(default_factory=list)
    channels: dict[str, dict[str, Any]] = field(default_factory=dict)
    threads: dict[tuple[str, str], list[dict[str, Any]]] = field(default_factory=dict)
    search_matches: list[dict[str, Any]] = field(default_factory=list)


_STATE = ApiState()


def _slack_response(data: dict[str, Any]) -> SlackResponse:
    """Build a real slack-sdk response so ``validate`` and ``get`` run as shipped."""
    return SlackResponse(
        client=None,
        http_verb="GET",
        api_url="https://slack.com/api/unit-test",
        req_args={},
        data=data,
        headers={},
        status_code=200,
    )


class _RecordingWebClient:
    """Stand-in for the HTTP send in slack-sdk ``WebClient.api_call``.

    ``conversations.replies`` forwards ``channel`` and ``ts`` and does not
    inspect channel type. This client records that call and returns the body
    configured for that exact channel and timestamp.
    """

    def __init__(self, token: str, timeout: int | None = None) -> None:
        del timeout
        self.token = token

    def _record(self, method: str, channel: str | None = None, ts: str | None = None) -> None:
        _STATE.calls.append(ApiCall(method=method, token=self.token, channel=channel, ts=ts))

    def conversations_replies(self, *, channel: str, ts: str, **kwargs: Any) -> SlackResponse:
        del kwargs
        self._record("conversations_replies", channel, ts)
        messages = _STATE.threads.get((channel, ts), [])
        return _slack_response({"ok": True, "messages": messages})

    def conversations_history(self, *, channel: str, **kwargs: Any) -> SlackResponse:
        del kwargs
        self._record("conversations_history", channel)
        return _slack_response({"ok": True, "messages": []})

    def conversations_info(self, *, channel: str, **kwargs: Any) -> SlackResponse:
        del kwargs
        self._record("conversations_info", channel)
        channel_info = _STATE.channels[channel]
        return _slack_response({"ok": True, "channel": channel_info})

    def conversations_list(self, **kwargs: Any) -> SlackResponse:
        del kwargs
        self._record("conversations_list")
        return _slack_response(
            {
                "ok": True,
                "channels": list(_STATE.channels.values()),
                "response_metadata": {"next_cursor": ""},
            }
        )

    def search_messages(self, **kwargs: Any) -> SlackResponse:
        del kwargs
        self._record("search_messages")
        return _slack_response({"ok": True, "messages": {"matches": list(_STATE.search_matches)}})

    def auth_test(self, **kwargs: Any) -> SlackResponse:
        del kwargs
        self._record("auth_test")
        return _slack_response({"ok": True, "team_id": "TPLACEHOLDER"})

    def users_profile_get(self, **kwargs: Any) -> SlackResponse:
        del kwargs
        self._record("users_profile_get")
        return _slack_response({"ok": True, "profile": {"real_name": "Owner"}})


def _channel(
    channel_id: str,
    name: str,
    *,
    is_im: bool = False,
    is_mpim: bool = False,
    is_private: bool = False,
) -> dict[str, Any]:
    return {
        "id": channel_id,
        "name": name,
        "is_im": is_im,
        "is_mpim": is_mpim,
        "is_private": is_private,
        "is_channel": not is_im and not is_mpim,
        "is_member": True,
    }


def _reset_workspace() -> ApiState:
    global _STATE
    _STATE = ApiState(
        channels={
            PUBLIC_CHANNEL: _channel(PUBLIC_CHANNEL, "general"),
            ASKING_PUBLIC_CHANNEL: _channel(ASKING_PUBLIC_CHANNEL, "random"),
            PRIVATE_CHANNEL: _channel(PRIVATE_CHANNEL, "exec-only", is_private=True),
            OTHER_PRIVATE_CHANNEL: _channel(OTHER_PRIVATE_CHANNEL, "legal-only", is_private=True),
            DM_CHANNEL: _channel(DM_CHANNEL, "directmessage", is_im=True),
            MPIM_CHANNEL: _channel(MPIM_CHANNEL, "mpdm-group", is_mpim=True, is_private=True),
            EXCLUDED_CHANNEL: _channel(EXCLUDED_CHANNEL, "secure-board"),
        }
    )
    _remember_thread(PUBLIC_CHANNEL, PUBLIC_SECRET)
    _remember_thread(PRIVATE_CHANNEL, PRIVATE_SECRET, PRIVATE_REPLY)
    _remember_thread(OTHER_PRIVATE_CHANNEL, OTHER_PRIVATE_SECRET)
    _remember_thread(DM_CHANNEL, DM_SECRET)
    _remember_thread(MPIM_CHANNEL, MPIM_SECRET)
    _remember_thread(EXCLUDED_CHANNEL, EXCLUDED_SECRET)
    return _STATE


def _remember_thread(channel_id: str, body: str, reply: str | None = None) -> None:
    messages = [{"text": body, "user": "UOWNER", "ts": THREAD_TS}]
    if reply is not None:
        messages.append({"text": reply, "user": "UREPLIER", "ts": REPLY_TS})
    _STATE.threads[(channel_id, THREAD_TS)] = messages


def _permalink(channel_id: str) -> str:
    return f"https://workspace.slack.com/archives/{channel_id}/p{RAW_TS}"


def _question(*channel_ids: str) -> str:
    links = " ".join(_permalink(channel_id) for channel_id in channel_ids)
    return f"Summarize {links}"


def _content_fetches(channel_id: str) -> list[ApiCall]:
    return [call for call in _STATE.calls if call.method in CONTENT_METHODS and call.channel == channel_id]


def _rendered(chunks: list[InferenceChunk]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        parts.append(chunk.content)
        parts.append(chunk.blurb)
        parts.append(chunk.semantic_identifier)
        parts.extend(chunk.match_highlights)
    return "\n".join(parts)


def _assert_not_read(channel_id: str, rendered: str, *secrets: str) -> None:
    fetches = _content_fetches(channel_id)
    assert fetches == [], f"{channel_id} was read before access was refused: {[(call.method, call.ts) for call in fetches]}"
    for secret in secrets:
        assert secret not in rendered, f"{secret} was returned for {channel_id}"


def _assert_read(channel_id: str, rendered: str, secret: str) -> None:
    fetches = _content_fetches(channel_id)
    assert fetches, f"{channel_id} was not read"
    assert all(call.token == ACCESS_TOKEN for call in fetches)
    reply_calls = [call for call in fetches if call.method == "conversations_replies"]
    assert reply_calls
    assert all(call.ts == THREAD_TS for call in reply_calls)
    assert secret in rendered


def _context(channel_type: ChannelType, channel_id: str) -> SlackContext:
    return SlackContext(
        channel_type=channel_type,
        channel_id=channel_id,
        user_id="UASKER",
        message_ts="1700000000.000100",
    )


def _search_settings() -> MagicMock:
    settings = MagicMock()
    settings.enable_contextual_rag = False
    settings.multipass_indexing = False
    settings.model_name = "unit-test-embedder"
    settings.provider_type = None
    return settings


def _run_bot_search(
    query: str,
    *,
    slack_context: SlackContext | None,
    entities: dict[str, Any],
    bot_token: str | None,
) -> list[InferenceChunk]:
    """Call the search tool method that the Slack bot uses for federated search."""
    tool = SearchTool(
        tool_id=1,
        emitter=None,
        user=MagicMock(),
        persona_search_info=PersonaSearchInfo(
            document_set_names=["slack-docs"],
            search_start_date=None,
            attached_document_ids=[],
            hierarchy_node_ids=[],
        ),
        llm=MagicMock(),
        document_index=MagicMock(),
        user_selected_filters=None,
        project_id_filter=None,
        slack_context=slack_context,
    )
    return tool._run_slack_search(
        query=query,
        access_token=ACCESS_TOKEN,
        bot_token=bot_token,
        entities=entities,
        search_settings=_search_settings(),
    )


class _CharTokenizer:
    def encode(self, string: str) -> list[int]:
        return [ord(char) for char in string]

    def tokenize(self, string: str) -> list[str]:
        return list(string)

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


@pytest.fixture
def slack_api() -> Iterator[ApiState]:
    state = _reset_workspace()
    llm = MagicMock()
    llm.config.model_name = "unit-test-model"
    llm.config.api_base = ""
    llm.config.model_provider = "unit-test"
    llm.invoke.side_effect = RuntimeError("llm disabled in unit test")

    embedder = MagicMock()
    embedder.embedding_model.tokenizer = _CharTokenizer()
    multipass = MultipassConfig(multipass_indexing=False, enable_large_chunks=False)
    redis_client = MagicMock()
    redis_client.get.return_value = None

    with (
        patch(
            "onyx.context.search.federated.slack_search.get_default_llm",
            return_value=llm,
        ),
        patch(
            "onyx.context.search.federated.slack_search.DefaultIndexingEmbedder.from_db_search_settings",
            return_value=embedder,
        ),
        patch(
            "onyx.context.search.federated.slack_search.get_multipass_config",
            return_value=multipass,
        ),
        patch(
            "onyx.context.search.federated.slack_search.ENABLE_CONTEXTUAL_RAG",
            False,
        ),
        patch(
            "onyx.context.search.federated.slack_search.WebClient",
            _RecordingWebClient,
        ),
        patch(
            "onyx.federated_connectors.slack.federated_connector.WebClient",
            _RecordingWebClient,
        ),
        patch(
            "onyx.context.search.federated.slack_search.get_redis_client",
            return_value=redis_client,
        ),
    ):
        yield state


def test_keyword_search_skips_private_channels_for_the_bot(
    slack_api: ApiState,
) -> None:
    """Control: keyword matches already go through the private-channel skip."""
    slack_api.search_matches = [
        {
            "text": PRIVATE_SECRET,
            "permalink": _permalink(PRIVATE_CHANNEL),
            "ts": THREAD_TS,
            "channel": {"id": PRIVATE_CHANNEL, "name": "exec-only"},
            "username": "owner",
            "score": 1.0,
        },
        {
            "text": PUBLIC_SECRET,
            "permalink": _permalink(PUBLIC_CHANNEL),
            "ts": THREAD_TS,
            "channel": {"id": PUBLIC_CHANNEL, "name": "general"},
            "username": "owner",
            "score": 0.5,
        },
    ]

    result = query_slack(
        query_string="performance",
        access_token=ACCESS_TOKEN,
        allowed_private_channel=None,
        bot_token=BOT_TOKEN,
        include_dm=False,
        entities={},
    )

    returned_channels = {message.channel_id for message in result.messages}
    assert PRIVATE_CHANNEL not in returned_channels
    assert PUBLIC_CHANNEL in returned_channels
    assert _content_fetches(PRIVATE_CHANNEL) == []


def test_public_channel_question_does_not_read_private_dm_or_group_dm_threads(
    slack_api: ApiState,
    caplog: pytest.LogCaptureFixture,
) -> None:
    del slack_api
    chunks = _run_bot_search(
        _question(PRIVATE_CHANNEL, DM_CHANNEL, MPIM_CHANNEL, PUBLIC_CHANNEL),
        slack_context=_context(ChannelType.PUBLIC_CHANNEL, ASKING_PUBLIC_CHANNEL),
        entities={},
        bot_token=BOT_TOKEN,
    )
    rendered = _rendered(chunks)

    assert "Slack federated search error" not in caplog.text
    _assert_read(PUBLIC_CHANNEL, rendered, PUBLIC_SECRET)
    _assert_not_read(PRIVATE_CHANNEL, rendered, PRIVATE_SECRET, PRIVATE_REPLY)
    _assert_not_read(DM_CHANNEL, rendered, DM_SECRET)
    _assert_not_read(MPIM_CHANNEL, rendered, MPIM_SECRET)


def test_direct_fetch_of_another_private_channel_is_refused(
    slack_api: ApiState,
) -> None:
    """The allowed channel is the one in the question, not a different private channel."""
    del slack_api
    chunks = _run_bot_search(
        _question(OTHER_PRIVATE_CHANNEL),
        slack_context=_context(ChannelType.PRIVATE_CHANNEL, PRIVATE_CHANNEL),
        entities={},
        bot_token=BOT_TOKEN,
    )
    rendered = _rendered(chunks)
    _assert_not_read(OTHER_PRIVATE_CHANNEL, rendered, OTHER_PRIVATE_SECRET)


def test_direct_fetch_allows_the_private_channel_where_the_question_was_asked(
    slack_api: ApiState,
) -> None:
    """Empty connector config lets the bot read the private channel it was asked in."""
    del slack_api
    chunks = _run_bot_search(
        _question(PRIVATE_CHANNEL),
        slack_context=_context(ChannelType.PRIVATE_CHANNEL, PRIVATE_CHANNEL),
        entities={},
        bot_token=BOT_TOKEN,
    )
    _assert_read(PRIVATE_CHANNEL, _rendered(chunks), PRIVATE_SECRET)


def test_direct_fetch_respects_excluded_channel_names(
    slack_api: ApiState,
) -> None:
    del slack_api
    chunks = _run_bot_search(
        _question(EXCLUDED_CHANNEL, PUBLIC_CHANNEL),
        slack_context=_context(ChannelType.PUBLIC_CHANNEL, ASKING_PUBLIC_CHANNEL),
        entities={"exclude_channels": ["secure-*"]},
        bot_token=BOT_TOKEN,
    )
    rendered = _rendered(chunks)
    _assert_read(PUBLIC_CHANNEL, rendered, PUBLIC_SECRET)
    _assert_not_read(EXCLUDED_CHANNEL, rendered, EXCLUDED_SECRET)


def test_include_flags_block_direct_fetches_without_a_bot_token(
    slack_api: ApiState,
) -> None:
    """Web federated search also passes team_id=None from the search tool."""
    del slack_api
    chunks = _run_bot_search(
        _question(PRIVATE_CHANNEL, DM_CHANNEL, MPIM_CHANNEL, PUBLIC_CHANNEL),
        slack_context=None,
        entities={
            "include_private_channels": False,
            "include_dm": False,
            "include_group_dm": False,
        },
        bot_token=None,
    )
    rendered = _rendered(chunks)
    _assert_read(PUBLIC_CHANNEL, rendered, PUBLIC_SECRET)
    _assert_not_read(PRIVATE_CHANNEL, rendered, PRIVATE_SECRET, PRIVATE_REPLY)
    _assert_not_read(DM_CHANNEL, rendered, DM_SECRET)
    _assert_not_read(MPIM_CHANNEL, rendered, MPIM_SECRET)


def test_connector_search_checks_the_url_channel_before_replies(
    slack_api: ApiState,
) -> None:
    """Team id enables the post-merge type filter only after replies has already run."""
    del slack_api
    connector = SlackFederatedConnector(
        {
            "client_id": CONNECTOR_CLIENT_ID,
            "client_secret": CONNECTOR_CLIENT_SECRET,
        }
    )
    session = MagicMock()
    session.__enter__.return_value = MagicMock()
    session.__exit__.return_value = False

    with (
        patch(
            "onyx.federated_connectors.slack.federated_connector.get_session_with_current_tenant",
            return_value=session,
        ),
        patch(
            "onyx.context.search.federated.slack_search.get_current_search_settings",
            return_value=_search_settings(),
        ),
    ):
        chunks = connector.search(
            query=ChunkIndexRequest(
                query=_question(PRIVATE_CHANNEL, EXCLUDED_CHANNEL, PUBLIC_CHANNEL),
                filters=IndexFilters(access_control_list=None),
            ),
            entities={
                "include_private_channels": True,
                "include_dm": True,
                "include_group_dm": True,
                "exclude_channels": ["secure-*"],
            },
            access_token=ACCESS_TOKEN,
            slack_event_context=_context(ChannelType.PUBLIC_CHANNEL, ASKING_PUBLIC_CHANNEL),
            bot_token=BOT_TOKEN,
        )

    rendered = _rendered(chunks)
    _assert_read(PUBLIC_CHANNEL, rendered, PUBLIC_SECRET)
    _assert_not_read(PRIVATE_CHANNEL, rendered, PRIVATE_SECRET, PRIVATE_REPLY)
    _assert_not_read(EXCLUDED_CHANNEL, rendered, EXCLUDED_SECRET)


def test_connector_search_respects_include_flags_before_replies(
    slack_api: ApiState,
) -> None:
    del slack_api
    connector = SlackFederatedConnector(
        {
            "client_id": CONNECTOR_CLIENT_ID,
            "client_secret": CONNECTOR_CLIENT_SECRET,
        }
    )
    session = MagicMock()
    session.__enter__.return_value = MagicMock()
    session.__exit__.return_value = False

    with (
        patch(
            "onyx.federated_connectors.slack.federated_connector.get_session_with_current_tenant",
            return_value=session,
        ),
        patch(
            "onyx.context.search.federated.slack_search.get_current_search_settings",
            return_value=_search_settings(),
        ),
    ):
        chunks = connector.search(
            query=ChunkIndexRequest(
                query=_question(PRIVATE_CHANNEL, DM_CHANNEL, PUBLIC_CHANNEL),
                filters=IndexFilters(access_control_list=None),
            ),
            entities={
                "include_private_channels": False,
                "include_dm": False,
                "include_group_dm": False,
            },
            access_token=ACCESS_TOKEN,
            slack_event_context=None,
            bot_token=None,
        )

    rendered = _rendered(chunks)
    _assert_read(PUBLIC_CHANNEL, rendered, PUBLIC_SECRET)
    _assert_not_read(PRIVATE_CHANNEL, rendered, PRIVATE_SECRET, PRIVATE_REPLY)
    _assert_not_read(DM_CHANNEL, rendered, DM_SECRET)
