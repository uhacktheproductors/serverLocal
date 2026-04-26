import os
import re
import json
import asyncio
import requests
import numpy as np
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from ddgs import DDGS

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

SYSTEM_PROMPT = """
You are an Football Data Analyst. When a user asks for a player based on a query (e.g., 'fastest left back under 23’):
	1.	Search: Query the attached Data Store and also by for players searching the web yourself for them from as many sources as you can find besides any EA FC or FIFA game related sites. Always search and respond with 5 players matching the search query no matter what the user requests. The price variable given by the user should always be enforced with the “price” data from the table depending on the users budget if they mention it. The RAG data should always be your first priority when searching for data and only use the web as a final last measure. Always respect what filters the user adds when searching for data such as age weight height etc..
	2.	Analyze: Compare the 'Expected Goals' (xG), 'Pass Completion', and 'Tackles Won' metrics across the top 5 results. If you’re not given a detailed search query continue anyways. Also try and see if you can find information about what other teams need the same playing position in their team and add that to the description. Also give for each player from the top 5 the current price for the player in the market, a rating out of 10, and a risk factor rating out of 10.
	3.	Recommend: Identify the 'Best Player' by weighting current form (last 10 matches) and potential, also always respond with specifically 5 players even if only one fits the best player criteria.
	4.	Justify: Explain why you chose them compared to the others in the description.

Send the response in json format no matter what including each of the 5 players with a string name field called “name” that has the player name without any additions and DONT use Player A, Player B, Player X, Player Y etc. in the name field, ALWAYS use the football player name. The justification should as “description” as a string, risk factor as “risk” as a number from 1-10, the price in Euro as “price” as a string formatted like 50m € or 50k €, the rating as “rating” as a number from 1-10, age as “age” as a number, height as “height” as text formatted like 1.83m, weight as “weight” as text formatted like 67kg, suitability for U-Cluj team as “suitability” as a number from 1-10, character as the personality of the player formatted in singular word characteristics separated by a comma and a space and only show a maximum of 3, positions as “positions” as a text meaning all positions the player plays with the most common one being the first one and separating all of them by a comma and a space, currentForm as “currentForm” as a number from 1-10 as a rating of his performance in the last season he played in.

This is the table format you should send back as json with the id being the number rating the player is, DONT add any extra tables such as recommendation:
create table public.players (
  id bigint not null,
  name text not null,
  description text not null,
  risk bigint null,
  price text null,
  rating bigint null,
  age bigint null,
  height text null,
  weight text null,
  suitability bigint null,
  character text null,
  positions text null,
  currentForm bigint null,
  constraint players_pkey primary key (id)
) TABLESPACE pg_default;
"""

DATA_DIR = Path("./data")
CACHE_DIR = Path("./.rag_cache")
CACHE_CHUNKS = CACHE_DIR / "chunks.json"
CACHE_EMBEDDINGS = CACHE_DIR / "embeddings.npy"
CHUNK_SIZE = 200

_embedder: SentenceTransformer | None = None
_chunks: list[str] = []
_embeddings: np.ndarray | None = None


def _cache_valid(data_files: list[Path]) -> bool:
    if not CACHE_CHUNKS.exists() or not CACHE_EMBEDDINGS.exists():
        return False
    cache_mtime = min(CACHE_CHUNKS.stat().st_mtime, CACHE_EMBEDDINGS.stat().st_mtime)
    return all(f.stat().st_mtime <= cache_mtime for f in data_files)


def _build_rag_index():
    global _embedder, _chunks, _embeddings
    DATA_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    data_files = [f for f in DATA_DIR.iterdir() if f.is_file()]
    if not data_files:
        print("No data files found in ./data — skipping RAG index.", flush=True)
        return

    print("Loading embedding model...", flush=True)
    _embedder = SentenceTransformer("all-MiniLM-L6-v2")

    if _cache_valid(data_files):
        print("Loading RAG index from cache...", flush=True)
        _chunks = json.loads(CACHE_CHUNKS.read_text())
        _embeddings = np.load(CACHE_EMBEDDINGS)
        print(f"RAG index loaded: {len(_chunks)} chunks.", flush=True)
        return

    all_text = []
    for f in data_files:
        try:
            all_text.append(f.read_text(encoding="utf-8").strip())
            print(f"Loaded data file: {f.name}", flush=True)
        except Exception as e:
            print(f"Skipping {f.name}: {e}", flush=True)
    combined = "\n\n".join(all_text)
    words = combined.split()
    _chunks = [" ".join(words[i:i + CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]
    print(f"Embedding {len(_chunks)} chunks...", flush=True)
    _embeddings = _embedder.encode(_chunks, convert_to_numpy=True, show_progress_bar=True, batch_size=64)
    _embeddings = _embeddings / np.linalg.norm(_embeddings, axis=1, keepdims=True)

    CACHE_DIR.mkdir(exist_ok=True)
    CACHE_CHUNKS.write_text(json.dumps(_chunks))
    np.save(CACHE_EMBEDDINGS, _embeddings)
    print(f"RAG index built and cached: {len(_chunks)} chunks from {len(data_files)} file(s).", flush=True)


RESPONSE_RESERVE = 512


def _count_tokens(text: str) -> int:
    return len(llm.tokenize(text.encode("utf-8"), add_bos=False))


def retrieve(query: str, budget: int = 0) -> str:
    if _embedder is None or _embeddings is None:
        return ""
    q = _embedder.encode([query], convert_to_numpy=True)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    scores = (q @ _embeddings.T)[0]
    ranked = np.argsort(scores)[::-1]
    selected, used = [], 0
    for i in ranked:
        chunk = _chunks[i]
        cost = _count_tokens(chunk)
        if used + cost > budget:
            break
        selected.append(chunk)
        used += cost
    return "\n\n".join(selected)


_build_rag_index()

MODEL_FILE = "gemma-4-E4B-it-Q4_K_M.gguf"
MODEL_URL = f"https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/{MODEL_FILE}"
MODEL_PATH = Path("./models") / MODEL_FILE


def download_model():
    MODEL_PATH.parent.mkdir(exist_ok=True)
    if MODEL_PATH.exists():
        print(f"Model already cached at {MODEL_PATH}", flush=True)
        return

    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    tmp_path = MODEL_PATH.with_suffix(".tmp")

    max_retries = 5
    for attempt in range(1, max_retries + 1):
        downloaded = tmp_path.stat().st_size if tmp_path.exists() else 0
        resume_headers = {**headers, "Range": f"bytes={downloaded}-"} if downloaded else headers

        print(f"Downloading {MODEL_FILE} (attempt {attempt}, resuming from {downloaded // 1024 // 1024} MB)...", flush=True)
        try:
            with requests.get(MODEL_URL, headers=resume_headers, stream=True, timeout=(10, 30)) as r:
                r.raise_for_status()
                total = downloaded + int(r.headers.get("content-length", 0))
                with open(tmp_path, "ab") as f:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded / total * 100
                            mb = downloaded / 1024 / 1024
                            total_mb = total / 1024 / 1024
                            print(f"  {mb:.0f} / {total_mb:.0f} MB ({pct:.1f}%)", flush=True)
            tmp_path.rename(MODEL_PATH)
            print("Download complete.", flush=True)
            return
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as e:
            print(f"Download interrupted: {e}. Retrying...", flush=True)

    raise RuntimeError(f"Failed to download model after {max_retries} attempts.")


download_model()

print("Loading model...", flush=True)
llm = Llama(
    model_path=str(MODEL_PATH),
    n_gpu_layers=-1,
    n_ctx=4096,
    n_batch=2048,
    n_threads=10,
    flash_attn=True,
    use_mlock=True,
    verbose=False,
)
print("Model ready.", flush=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


def web_search(query: str, max_results: int = 3) -> str:
    print(f"[web] searching: {query!r}", flush=True)
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    for r in results:
        print(f"[web]   {r.get('href', 'unknown')}  —  {r['title']}", flush=True)
    return "\n\n".join(f"{r['title']}\n{r['body'][:300]}" for r in results) if results else ""


def extract_json(content: str) -> str:
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    content = re.sub(r"```(?:json)?\s*", "", content).replace("```", "")
    match = re.search(r"(\[.*\]|\{.*\})", content, flags=re.DOTALL)
    return match.group(0).strip() if match else content.strip()


async def build_messages(user_message: str) -> tuple[list[dict], int]:
    import time
    t0 = time.perf_counter()
    search_results, rag_context_raw = await asyncio.gather(
        asyncio.to_thread(web_search, user_message),
        asyncio.to_thread(retrieve, user_message, 2000),
    )
    retrieval_ms = int((time.perf_counter() - t0) * 1000)
    print(f"[timing] web+rag parallel: {retrieval_ms}ms", flush=True)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if search_results:
        messages.append({"role": "system", "content": f"Web search results (use as supplementary):\n{search_results}"})
    if rag_context_raw:
        messages.append({"role": "system", "content": f"PRIORITY — Data Store entries (prefer these over web results):\n{rag_context_raw}"})
    messages.append({"role": "user", "content": user_message})

    total_tokens = sum(_count_tokens(m["content"]) for m in messages)
    print(f"[timing] prompt tokens: {total_tokens}", flush=True)
    return messages, retrieval_ms


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    import time
    print(f"[request] {request.message!r}", flush=True)
    messages, retrieval_ms = await build_messages(request.message)
    t1 = time.perf_counter()
    result = await asyncio.to_thread(
        llm.create_chat_completion,
        messages=messages,
        max_tokens=1500,
        temperature=0.7,
        top_p=0.9,
        response_format={"type": "json_object"},
    )
    llm_ms = int((time.perf_counter() - t1) * 1000)
    print(f"[timing] llm inference: {llm_ms}ms  |  total: {retrieval_ms + llm_ms}ms", flush=True)
    response = extract_json(result["choices"][0]["message"]["content"] or "")
    print(f"[response] {response}", flush=True)
    return ChatResponse(response=response)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
