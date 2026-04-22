from __future__ import annotations
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from src.config import settings
from src.generate import Generator
from src.retrieve import Retriever


# minimal logger for both local debugging and viewing in Cloud Run's log panel
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("exile_oracle")


# request / response models 
class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="The user's question.")
    top_k: int | None = Field(
        None,
        ge=1,
        le=20,
        description="Optional override for number of chunks to retrieve (default from settings).",
    )


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    usage: dict
    latency_ms: int



# the app-level state dict holds long-lived objects (Retriever, Generator)
# that we want to build once at startup and reuse across requests
app_state: dict = {}


# runs once at startup to initialize retriever and generator and once at shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: building Retriever and Generator...")
    app_state["retriever"] = Retriever()
    app_state["generator"] = Generator()
    logger.info("Startup complete.")
    yield # code after yield runs at shutdown, nothing to cleanup for now
    logger.info("Shutting down.")


app = FastAPI(
    title="Exile Oracle",
    description="RAG assistant for Path of Exile (PoE 1) backed by the PoE Wiki.",
    version=settings.app_version,
    lifespan=lifespan,
)


# ROUTES

# liveness / readiness probe endpoint
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": settings.app_version}


# main RAG endpoint: retrieve relevant chunks, then generate a grounded answer
@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    t0 = time.monotonic()
    retriever: Retriever = app_state["retriever"]
    generator: Generator = app_state["generator"]

    try:
        chunks = retriever.retrieve(request.query, top_k=request.top_k)
    except Exception as e:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=502, detail=f"Retrieval error: {e}")

    try:
        answer = generator.generate(request.query, chunks)
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(status_code=502, detail=f"Generation error: {e}")

    latency_ms = int((time.monotonic() - t0) * 1000)
    logger.info(
        "ask: query=%r top_k=%d chunks=%d in_tok=%d out_tok=%d latency_ms=%d",
        request.query,
        request.top_k or settings.top_k,
        len(chunks),
        answer.prompt_tokens,
        answer.completion_tokens,
        latency_ms,
    )

    return AskResponse(
        answer=answer.answer,
        sources=[s.as_dict() for s in answer.sources],
        usage={
            "prompt_tokens": answer.prompt_tokens,
            "completion_tokens": answer.completion_tokens,
        },
        latency_ms=latency_ms,
    )


# static front end
INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Exile Oracle</title>
<style>
  :root {
    --fg: #d4d4d4;
    --fg-dim: #8a8a8a;
    --bg: #1a1a1a;
    --panel: #242424;
    --border: #3a3a3a;
    --accent: #c9a76f;
    --ok: #8bb26a;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0;
    background: var(--bg);
    color: var(--fg);
    font-family: Consolas, "Courier New", Menlo, monospace;
    font-size: 14px;
    line-height: 1.5;
  }
  main {
    max-width: 860px;
    margin: 48px auto;
    padding: 0 24px;
  }
  h1 {
    font-size: 18px;
    font-weight: 600;
    margin: 0 0 4px 0;
    letter-spacing: 0.02em;
  }
  .subtitle {
    color: var(--fg-dim);
    margin-bottom: 32px;
    font-size: 13px;
  }
  form {
    display: flex;
    gap: 8px;
    margin-bottom: 24px;
  }
  input[type="text"] {
    flex: 1;
    background: var(--panel);
    color: var(--fg);
    border: 1px solid var(--border);
    padding: 10px 12px;
    font-family: inherit;
    font-size: 14px;
    outline: none;
  }
  input[type="text"]:focus {
    border-color: var(--accent);
  }
  button {
    background: var(--panel);
    color: var(--fg);
    border: 1px solid var(--border);
    padding: 10px 18px;
    font-family: inherit;
    font-size: 14px;
    cursor: pointer;
  }
  button:hover:not(:disabled) {
    border-color: var(--accent);
    color: var(--accent);
  }
  button:disabled {
    opacity: 0.5;
    cursor: wait;
  }
  .panel {
    background: var(--panel);
    border: 1px solid var(--border);
    padding: 16px;
    margin-bottom: 16px;
    white-space: pre-wrap;
  }
  .panel-label {
    color: var(--fg-dim);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
  }
  .sources {
    font-size: 13px;
    color: var(--fg-dim);
  }
  .sources .source {
    padding: 4px 0;
    border-top: 1px solid var(--border);
  }
  .sources .source:first-of-type {
    border-top: none;
  }
  .source-title {
    color: var(--fg);
  }
  .source-meta {
    color: var(--fg-dim);
    font-size: 12px;
  }
  .meta {
    color: var(--fg-dim);
    font-size: 12px;
    margin-top: 8px;
  }
  .error {
    color: #c57070;
  }
  .examples {
    font-size: 13px;
    color: var(--fg-dim);
    margin-bottom: 24px;
  }
  .examples a {
    color: var(--fg-dim);
    text-decoration: underline;
    cursor: pointer;
    margin-right: 14px;
  }
  .examples a:hover {
    color: var(--accent);
  }
  a.docs-link {
    color: var(--fg-dim);
  }
  a.docs-link:hover {
    color: var(--accent);
  }
</style>
</head>
<body>
<main>
  <h1>exile-oracle</h1>
  <div class="subtitle">
    Ask questions about Path of Exile (PoE 1) mechanics, skills, items, and keystones.
    Answers are grounded in the PoE Wiki. <a class="docs-link" href="/docs">API docs</a>
  </div>

<div class="examples">
    try:
    <a data-q="What does the Acrobatics keystone do?">Acrobatics</a>
    <a data-q="What is the difference between more and increased damage?">More vs increased</a>
    <a data-q="What are some minion skills I can use early game?">Early minions</a>
    <a data-q="What does Headhunter do?">Headhunter</a>
  </div>

  <form id="ask-form">
    <input id="q" type="text" placeholder="Ask a question..." autocomplete="off" required>
    <button id="submit-btn" type="submit">ask</button>
  </form>

  <div id="result" style="display:none">
    <div class="panel">
      <div class="panel-label">answer</div>
      <div id="answer"></div>
    </div>
    <div class="panel">
      <div class="panel-label">sources</div>
      <div id="sources" class="sources"></div>
    </div>
    <div id="meta" class="meta"></div>
  </div>

  <div id="error" class="error" style="display:none"></div>
</main>

<script>
  const form = document.getElementById('ask-form');
  const input = document.getElementById('q');
  const btn = document.getElementById('submit-btn');
  const result = document.getElementById('result');
  const answerEl = document.getElementById('answer');
  const sourcesEl = document.getElementById('sources');
  const metaEl = document.getElementById('meta');
  const errorEl = document.getElementById('error');

  // Clickable example queries.
  document.querySelectorAll('.examples a').forEach(el => {
    el.addEventListener('click', (e) => {
      e.preventDefault();
      input.value = el.getAttribute('data-q');
      form.dispatchEvent(new Event('submit'));
    });
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (!query) return;

    btn.disabled = true;
    btn.textContent = 'thinking...';
    errorEl.style.display = 'none';
    result.style.display = 'none';

    try {
      const res = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(err.detail || 'Request failed');
      }
      const data = await res.json();

      answerEl.textContent = data.answer;

      sourcesEl.innerHTML = '';
      data.sources.forEach(s => {
        const div = document.createElement('div');
        div.className = 'source';
        const title = document.createElement('div');
        title.className = 'source-title';
        title.textContent = s.source_title;
        const meta = document.createElement('div');
        meta.className = 'source-meta';
        meta.textContent = `category: ${s.category}   score: ${s.score.toFixed(3)}`;
        div.appendChild(title);
        div.appendChild(meta);
        sourcesEl.appendChild(div);
      });

      metaEl.textContent = `${data.usage.prompt_tokens} input tokens / ${data.usage.completion_tokens} output tokens / ${data.latency_ms} ms`;
      result.style.display = 'block';
    } catch (err) {
      errorEl.textContent = 'error: ' + err.message;
      errorEl.style.display = 'block';
    } finally {
      btn.disabled = false;
      btn.textContent = 'ask';
    }
  });
</script>
</body>
</html>
"""

# serve the static front end at the root URL
@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML)