import os
import re
import requests
import numpy as np
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from duckduckgo_search import DDGS

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

SYSTEM_PROMPT = """
You are an elite Football Data Analyst. When a user asks for a player based on a query (e.g., 'fastest left back under 23’):
	1.	Search: Query the attached Data Store for players searching the web yourself for them from as many sources as you can find.
	2.	Analyze: Compare the 'Expected Goals' (xG), 'Pass Completion', and 'Tackles Won' metrics across the top 5 results. If you’re not given a detailed search query continue anyways. Also try and see if you can find information about what other teams need the same playing position in their team and add that to the description. Also give for each player from the top 5 the current price for the player in the market, a rating out of 5, and a risk factor rating out of 5.
	3.	Recommend: Identify the 'Best Player' by weighting current form (last 10 matches) and potential.
	4.	Justify: Explain why you chose them compared to the others.
Send the response in json format no matter what including each of the 5 players with a string name field called “name” that has the player name without any additions. The justification should be in Romanian as “description” as a string, risk factor as “risk” as a number from 1-10, the price in Euro in Romanian as “price” as a string formatted like 50M € or 50k €, the rating as “rating” as a number from 1-10, height as “height” as text formatted like 1.83m, weight as “weight” as text formatted like 67kg, suitability for U-Cluj team as “suitability” as a number from 1-10, character as the personality of the player formatted in singular word characteristics separated by a comma and a space and only show a maximum of 3, positions as “positions” as a text meaning all positions the player plays with the most common one being the first one and separating all of them by a comma and a space, currentForm as “currentForm” as a number from 1-10 as a rating of his performance in the last season he played in, awayLost awayWon awayDraw homeLost homeWon homeDraw should all be as numbers.
This is the table format you should send back as json with the id being the number rating the player is: create table public.players (
  id bigint not null,
  name text not null,
  description text not null,
  risk bigint null,
  price text null,
  rating bigint null,
  height text null,
  weight text null,
  suitability bigint null,
  character text null,
  positions text null,
  currentForm bigint null,
  awayLost double precision null,
  awayWon double precision null,
  awayDraw double precision null,
  homeLost double precision null,
  homeWon double precision null,
  homeDraw double precision null,
  constraint players_pkey primary key (id)
) TABLESPACE pg_default;
"""

DATA_DIR = Path("./data")
CHUNK_SIZE = 200
TOP_K = 5

_embedder: SentenceTransformer | None = None
_chunks: list[str] = []
_embeddings: np.ndarray | None = None


def _build_rag_index():
    global _embedder, _chunks, _embeddings
    if not DATA_DIR.exists():
        return
    files = list(DATA_DIR.iterdir())
    if not files:
        return
    all_text = []
    for f in files:
        if f.is_file():
            try:
                all_text.append(f.read_text(encoding="utf-8").strip())
                print(f"Loaded data file: {f.name}", flush=True)
            except Exception as e:
                print(f"Skipping {f.name}: {e}", flush=True)
    combined = "\n\n".join(all_text)
    words = combined.split()
    _chunks = [" ".join(words[i:i + CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]
    print("Loading embedding model...", flush=True)
    _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    _embeddings = _embedder.encode(_chunks, convert_to_numpy=True, show_progress_bar=False)
    _embeddings = _embeddings / np.linalg.norm(_embeddings, axis=1, keepdims=True)
    print(f"RAG index built: {len(_chunks)} chunks from {len(files)} file(s).", flush=True)


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
    n_ctx=8192,
    flash_attn=True,
    verbose=False,
)
print("Model ready.", flush=True)

app = FastAPI()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


def web_search(query: str, max_results: int = 5) -> str:
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return "\n\n".join(f"{r['title']}\n{r['body']}" for r in results) if results else "No results found."


def extract_json(content: str) -> str:
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    content = re.sub(r"```(?:json)?\s*", "", content).replace("```", "")
    match = re.search(r"(\[.*\]|\{.*\})", content, flags=re.DOTALL)
    return match.group(0).strip() if match else content.strip()


def build_messages(user_message: str) -> list[dict]:
    search_results = web_search(user_message)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if search_results:
        messages.append({"role": "system", "content": f"Web search results:\n{search_results}"})
    fixed_tokens = sum(_count_tokens(m["content"]) for m in messages) + _count_tokens(user_message)
    context_budget = llm.n_ctx() - fixed_tokens - RESPONSE_RESERVE
    context = retrieve(user_message, budget=context_budget)
    if context:
        messages.append({"role": "system", "content": f"Relevant Data Store entries:\n{context}"})
    messages.append({"role": "user", "content": user_message})
    return messages


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    result = llm.create_chat_completion(
        messages=build_messages(request.message),
        max_tokens=2048,
        temperature=0.7,
        top_p=0.9,
        response_format={"type": "json_object"},
    )
    return ChatResponse(response=extract_json(result["choices"][0]["message"]["content"] or ""))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
