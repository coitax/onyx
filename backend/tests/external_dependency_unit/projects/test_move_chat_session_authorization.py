"""Authorization for moving a chat into a project.

These tests describe the required behavior of
``POST /user/projects/{project_id}/move_chat_session`` and of the chat turn
that later reads that session's project instructions.

They fail while a caller who does not own the target project can attach their
chat to it, and while a session linked to someone else's project still feeds
that project's instructions to the model.
"""

from collections.abc import Iterator
from queue import Queue
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from onyx.auth.schemas import UserRole
from onyx.auth.users import current_user
from onyx.chat.chat_state import ChatStateContainer
from onyx.chat.chat_utils import create_chat_session_from_request
from onyx.chat.chat_utils import get_custom_agent_prompt
from onyx.chat.emitter import Emitter
from onyx.chat.llm_loop import run_llm_loop
from onyx.chat.models import ChatMessageSimple
from onyx.chat.models import ExtractedContextFiles
from onyx.configs.constants import DEFAULT_PERSONA_ID
from onyx.configs.constants import MessageType
from onyx.db.chat import create_chat_session
from onyx.db.chat import get_chat_session_by_id
from onyx.db.engine.sql_engine import get_session
from onyx.db.engine.sql_engine import get_sqlalchemy_engine
from onyx.db.engine.sql_engine import SqlEngine
from onyx.db.enums import AccountType
from onyx.db.enums import Permission
from onyx.db.memory import UserInfo
from onyx.db.memory import UserMemoryContext
from onyx.db.models import Base
from onyx.db.models import ChatSession
from onyx.db.models import Persona
from onyx.db.models import User
from onyx.db.models import UserProject
from onyx.error_handling.exceptions import register_onyx_exception_handlers
from onyx.llm.interfaces import LLM
from onyx.llm.interfaces import LLMConfig
from onyx.llm.model_response import Delta
from onyx.llm.model_response import ModelResponseStream
from onyx.llm.model_response import StreamingChoice
from onyx.llm.models import LanguageModelInput
from onyx.server.features.projects.api import router as projects_router
from onyx.server.query_and_chat.models import ChatSessionCreationRequest
from onyx.server.query_and_chat.streaming_models import Packet

# Tables touched when the move endpoint runs and when a chat turn eager-loads
# the session persona (tools, files, document sets, hierarchy) plus its project.
_TABLE_NAMES = (
    "user",
    "oauth_account",
    "llm_provider",
    "model_configuration",
    "oauth_config",
    "mcp_server",
    "mcp_connection_config",
    "persona",
    "user_project",
    "chat_session",
    "tool",
    "persona__tool",
    "user_file",
    "persona__user_file",
    "document_set",
    "persona__document_set",
    "document",
    "persona__document",
    "hierarchy_node",
    "persona__hierarchy_node",
)
_TABLES = [Base.metadata.tables[name] for name in _TABLE_NAMES]


class _RecordingLLM(LLM):
    """Records the prompt the chat loop passes to ``LLM.stream`` and returns one token."""

    def __init__(self) -> None:
        self.prompts: list[LanguageModelInput] = []

    @property
    def config(self) -> LLMConfig:
        return LLMConfig(
            model_provider="mock",
            model_name="mock",
            temperature=0.0,
            max_input_tokens=1_000_000,
        )

    def stream(
        self,
        prompt: LanguageModelInput,
        tools: list[dict] | None = None,
        tool_choice: object = None,
        structured_response_format: dict | None = None,
        timeout_override: int | None = None,
        max_tokens: int | None = None,
        reasoning_effort: object = None,
        user_identity: object = None,
    ) -> Iterator[ModelResponseStream]:
        del (
            tools,
            tool_choice,
            structured_response_format,
            timeout_override,
            max_tokens,
            reasoning_effort,
            user_identity,
        )
        self.prompts.append(prompt)
        yield ModelResponseStream(
            id="recording",
            created="0",
            choice=StreamingChoice(
                finish_reason="stop",
                index=0,
                delta=Delta(content="ok"),
            ),
        )


def _prompt_text(prompt: LanguageModelInput) -> str:
    messages = prompt if isinstance(prompt, list) else [prompt]
    chunks: list[str] = []
    for message in messages:
        content = message.content
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for part in content:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    chunks.append(text)
    return "\n".join(chunks)


@pytest.fixture(scope="module")
def engine() -> Engine:
    SqlEngine.init_engine(pool_size=5, max_overflow=2)
    bound = get_sqlalchemy_engine()
    Base.metadata.create_all(bound, tables=_TABLES)
    return bound


@pytest.fixture()
def db_session(engine: Engine) -> Iterator[Session]:
    del engine
    with Session(get_sqlalchemy_engine(), expire_on_commit=False) as session:
        _ensure_default_persona(session)
        yield session


def _ensure_default_persona(db_session: Session) -> Persona:
    persona = db_session.get(Persona, DEFAULT_PERSONA_ID)
    if persona is None:
        persona = Persona(
            id=DEFAULT_PERSONA_ID,
            name="default-agent",
            description="default agent",
            builtin_persona=True,
            system_prompt=None,
            replace_base_system_prompt=False,
            deleted=False,
        )
        db_session.add(persona)
        db_session.commit()
    return persona


def _create_user(db_session: Session, *, basic_access: bool) -> User:
    user = User(
        id=uuid4(),
        email=f"move-auth-{uuid4().hex}@example.com",
        hashed_password="not-a-password",
        is_active=True,
        is_superuser=False,
        is_verified=True,
        role=UserRole.BASIC,
        account_type=AccountType.STANDARD,
        effective_permissions=(
            [Permission.BASIC_ACCESS.value] if basic_access else []
        ),
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def _create_project(
    db_session: Session, owner: User, name: str, instructions: str
) -> UserProject:
    project = UserProject(
        name=name,
        user_id=owner.id,
        instructions=instructions,
    )
    db_session.add(project)
    db_session.commit()
    db_session.refresh(project)
    return project


def _create_owned_chat(db_session: Session, owner: User) -> ChatSession:
    return create_chat_session(
        db_session=db_session,
        description="attacker chat",
        user_id=owner.id,
        persona_id=DEFAULT_PERSONA_ID,
        project_id=None,
    )


def _client_for(user: User, db_session: Session) -> TestClient:
    application = FastAPI()
    register_onyx_exception_handlers(application)
    application.include_router(projects_router)

    def _override_session() -> Iterator[Session]:
        yield db_session

    application.dependency_overrides[get_session] = _override_session
    application.dependency_overrides[current_user] = lambda: user
    return TestClient(application, raise_server_exceptions=False)


def _move(client: TestClient, project_id: int, chat_session_id: object) -> Response:
    return client.post(
        f"/user/projects/{project_id}/move_chat_session",
        json={"chat_session_id": str(chat_session_id)},
    )


def _persisted_project_id(chat_session_id: object) -> int | None:
    with Session(get_sqlalchemy_engine()) as fresh:
        row = fresh.get(ChatSession, chat_session_id)
        assert row is not None
        return row.project_id


def _model_prompt_text(db_session: Session, chat: ChatSession, owner: User) -> str:
    """Run the same load, prompt selection, and LLM call a chat turn performs."""
    loaded = get_chat_session_by_id(
        chat_session_id=chat.id,
        user_id=owner.id,
        db_session=db_session,
        eager_load_persona=True,
    )
    custom_agent_prompt = get_custom_agent_prompt(loaded.persona, loaded)
    recorder = _RecordingLLM()
    merged: Queue[tuple[int, Packet | Exception | object]] = Queue()
    run_llm_loop(
        emitter=Emitter(merged_queue=merged),
        state_container=ChatStateContainer(),
        simple_chat_history=[
            ChatMessageSimple(
                message="Repeat the instructions you were given.",
                token_count=8,
                message_type=MessageType.USER,
            )
        ],
        tools=[],
        custom_agent_prompt=custom_agent_prompt,
        context_files=ExtractedContextFiles(
            file_texts=[],
            image_files=[],
            use_as_search_filter=False,
            total_token_count=0,
            file_metadata=[],
            uncapped_token_count=0,
        ),
        persona=loaded.persona,
        user_memory_context=UserMemoryContext(user_info=UserInfo()),
        llm=recorder,
        token_counter=len,
        db_session=db_session,
    )
    assert recorder.prompts, "the chat loop never called the model"
    return "\n".join(_prompt_text(prompt) for prompt in recorder.prompts)


def test_moving_chat_into_another_users_project_does_not_link_or_leak_instructions(
    db_session: Session,
) -> None:
    victim_secret = f"victim-project-instructions-{uuid4().hex}"
    attacker_instructions = f"attacker-project-instructions-{uuid4().hex}"
    victim = _create_user(db_session, basic_access=True)
    attacker = _create_user(db_session, basic_access=True)
    victim_project = _create_project(
        db_session, victim, "victim-project", victim_secret
    )
    # The caller owns a different project. A check that only asks whether the
    # caller owns some project, then assigns the path id, must still fail.
    _create_project(db_session, attacker, "attacker-project", attacker_instructions)
    chat = _create_owned_chat(db_session, attacker)

    response = _move(_client_for(attacker, db_session), victim_project.id, chat.id)

    persisted_project_id = _persisted_project_id(chat.id)
    prompt = _model_prompt_text(db_session, chat, attacker)
    assert response.status_code == 404, (
        f"status={response.status_code} project_id={persisted_project_id} "
        f"victim_instructions_in_model_prompt={victim_secret in prompt}"
    )
    assert persisted_project_id is None
    assert victim_secret not in prompt
    assert attacker_instructions not in prompt
    with Session(get_sqlalchemy_engine()) as fresh:
        stored = fresh.get(ChatSession, chat.id)
        assert stored is not None
        assert stored.user_id == attacker.id
        victim_row = fresh.get(UserProject, victim_project.id)
        assert victim_row is not None
        assert victim_row.user_id == victim.id
        assert victim_row.instructions == victim_secret
        assert (
            fresh.query(ChatSession)
            .filter(ChatSession.project_id == victim_project.id)
            .count()
            == 0
        )


def test_session_linked_to_another_users_project_does_not_send_those_instructions(
    db_session: Session,
) -> None:
    """The project read at prompt-build time must be owned by the session owner.

    Authorizing a different project first, then pointing the session at the
    victim project, must not deliver the victim instructions. A check against
    one project id that then reads another project's instructions fails here.
    """
    victim_secret = f"victim-project-instructions-{uuid4().hex}"
    attacker_instructions = f"attacker-project-instructions-{uuid4().hex}"
    victim = _create_user(db_session, basic_access=True)
    attacker = _create_user(db_session, basic_access=True)
    victim_project = _create_project(
        db_session, victim, "victim-project", victim_secret
    )
    attacker_project = _create_project(
        db_session, attacker, "attacker-project", attacker_instructions
    )
    chat = _create_owned_chat(db_session, attacker)

    authorized = _move(
        _client_for(attacker, db_session), attacker_project.id, chat.id
    )
    assert authorized.status_code == 204, authorized.text

    # The session now points at a project that was not the one just authorized.
    stored = db_session.get(ChatSession, chat.id)
    assert stored is not None
    stored.project_id = victim_project.id
    db_session.commit()

    prompt = _model_prompt_text(db_session, chat, attacker)
    assert victim_secret not in prompt
    assert attacker_instructions not in prompt

    with Session(get_sqlalchemy_engine()) as fresh:
        row = fresh.get(ChatSession, chat.id)
        assert row is not None
        # Refusal must not unlink, delete, or reassign the chat as a side effect.
        assert row.project_id == victim_project.id
        assert row.user_id == attacker.id


def test_moving_chat_into_callers_own_project_uses_only_that_projects_instructions(
    db_session: Session,
) -> None:
    victim_secret = f"victim-project-instructions-{uuid4().hex}"
    attacker_instructions = f"attacker-project-instructions-{uuid4().hex}"
    victim = _create_user(db_session, basic_access=True)
    attacker = _create_user(db_session, basic_access=True)
    _create_project(db_session, victim, "victim-project", victim_secret)
    attacker_project = _create_project(
        db_session, attacker, "attacker-project", attacker_instructions
    )
    chat = _create_owned_chat(db_session, attacker)

    response = _move(
        _client_for(attacker, db_session), attacker_project.id, chat.id
    )

    assert response.status_code == 204, response.text
    assert _persisted_project_id(chat.id) == attacker_project.id
    prompt = _model_prompt_text(db_session, chat, attacker)
    assert attacker_instructions in prompt
    assert victim_secret not in prompt


def test_cannot_move_another_users_chat(
    db_session: Session,
) -> None:
    victim = _create_user(db_session, basic_access=True)
    attacker = _create_user(db_session, basic_access=True)
    project = _create_project(
        db_session, attacker, "attacker-project", f"own-{uuid4().hex}"
    )
    victim_chat = _create_owned_chat(db_session, victim)

    response = _move(_client_for(attacker, db_session), project.id, victim_chat.id)

    assert response.status_code == 404, response.text
    assert _persisted_project_id(victim_chat.id) is None


def test_user_without_basic_access_cannot_move_a_chat(
    db_session: Session,
) -> None:
    owner = _create_user(db_session, basic_access=False)
    project = _create_project(db_session, owner, "own-project", f"own-{uuid4().hex}")
    chat = _create_owned_chat(db_session, owner)

    response = _move(_client_for(owner, db_session), project.id, chat.id)

    assert response.status_code == 403, response.text
    assert _persisted_project_id(chat.id) is None


def test_creating_a_chat_in_another_users_project_is_rejected(
    db_session: Session,
) -> None:
    victim_secret = f"victim-project-instructions-{uuid4().hex}"
    victim = _create_user(db_session, basic_access=True)
    attacker = _create_user(db_session, basic_access=True)
    victim_project = _create_project(
        db_session, victim, "victim-project", victim_secret
    )
    before = db_session.query(ChatSession).count()

    with pytest.raises(ValueError, match="does not have access to project"):
        create_chat_session_from_request(
            chat_session_request=ChatSessionCreationRequest(
                persona_id=DEFAULT_PERSONA_ID,
                description="should not be stored",
                project_id=victim_project.id,
            ),
            user_id=attacker.id,
            db_session=db_session,
        )

    db_session.rollback()
    assert db_session.query(ChatSession).count() == before


def test_move_to_missing_project_does_not_link_chat(db_session: Session) -> None:
    attacker = _create_user(db_session, basic_access=True)
    chat = _create_owned_chat(db_session, attacker)
    chat_id = chat.id
    missing_id = 2_000_000_000

    response = _move(_client_for(attacker, db_session), missing_id, chat_id)
    # The current handler assigns the id and lets the foreign key abort the
    # transaction. Roll back so the follow-up read is not stuck on that error.
    db_session.rollback()

    persisted_project_id = _persisted_project_id(chat_id)
    assert response.status_code == 404, (
        f"status={response.status_code} project_id={persisted_project_id} "
        f"body={response.text}"
    )
    assert persisted_project_id is None
