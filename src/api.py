"""FastAPI service exposing the knowledge base QA agent over HTTP.

The embedder and QAAgent are built once at startup and reused across
requests -- the CLI's `ask` command pays that cost on every invocation, this
service pays it once. Run from src/:
    uvicorn api:app --reload
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from utils.embedder import Embedder
from utils.postgres_search_index import PostgresSearchIndex
from utils.qa_agent import QAAgent

_agent: QAAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    # No attach_context given, so this follows settings.attach_search_context
    # (off by default -- see PostgresSearchIndex's docstring) the same way the
    # CLI does when its own --attach-context flag isn't passed.
    _agent = QAAgent(PostgresSearchIndex(embedder=Embedder()))
    yield
    _agent = None


app = FastAPI(title="Knowledge base QA API", lifespan=lifespan)


def get_agent() -> QAAgent:
    assert _agent is not None, "QAAgent not initialised -- lifespan startup did not run"
    return _agent


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, agent: QAAgent = Depends(get_agent)) -> AskResponse:
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")
    return AskResponse(answer=agent.answer(request.question))
