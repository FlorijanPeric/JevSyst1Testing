"""Email router as a small web API, so each email doesn't pay Python startup + Qdrant connect (~1 s).

    ./run.sh serve                      # http://127.0.0.1:8090, only reachable on this machine
    curl -s localhost:8090/route -H 'Content-Type: application/json' -d '{"email": "The 3D printer is jammed"}'

Endpoints:
    GET  /health              profiles loaded, Qdrant in use
    POST /route   {email, learn?}       decision + answers; logged like the CLI (source "api")
    POST /fix     {email, person_id}    correct a decision (cache + real label + profile)
    POST /ok      {email}               confirm the last decision was right
    GET  /summary?days=7      text overview of logged decisions
"""
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient

import config
import settings
from profiles import get_store
from route_email import (PATH_LABELS, confirm, describe, final_id, fix, load_questions, log_decision, route)
from summary import summarize

app = FastAPI(title="Email router")
client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
store = get_store(client)
questions = load_questions()
started = time.time()


class RouteIn(BaseModel):
    email: str
    learn: bool = True


class FixIn(BaseModel):
    email: str
    person_id: str


class OkIn(BaseModel):
    email: str


@app.get("/health")
def health() -> dict:
    store.refresh()
    return {"ok": True, "profiles": len(store.profiles), "qdrant": config.QDRANT_URL,
            "model": config.CHAT_MODEL, "uptime_s": round(time.time() - started)}


@app.post("/route")
def route_email(body: RouteIn) -> dict:
    if not body.email.strip():
        raise HTTPException(400, "email is empty")
    args = settings.as_args(learn=body.learn)  # re-read routing.toml, so edits apply without a restart
    result = route(client, body.email, args, questions=questions)
    log_decision(body.email, result, source="api")
    person = result["person"]
    return {
        "forward_to": final_id(result),
        "name": person["name"] if person else None,
        "email": person["email"] if person else None,
        "summary": describe(result),
        "decided_by": PATH_LABELS[result["path"]],
        "path": result["path"],
        "auto_forward": result.get("auto", False),  # safe to forward without a human (decide.safe_to_auto)
        "answers": result["answers"],
        "verdict": result["verdict"],
        "candidates": [{"id": p["person_id"], "score": round(p["score"], 4)} for p in result["people"]],
        "ms": round(result["timer"].total),
        "timings_ms": {k: round(v) for k, v in result["timer"].stages.items()},
    }


@app.post("/fix")
def fix_email(body: FixIn) -> dict:
    message = fix(client, body.email, body.person_id)
    if message.startswith("Unknown person"):
        raise HTTPException(400, message)
    return {"message": message}


@app.post("/ok")
def ok_email(body: OkIn) -> dict:
    return {"message": confirm(body.email, client)}


@app.get("/summary", response_class=PlainTextResponse)
def summary(days: int = 7) -> str:
    return summarize(days)
