"""
agent_brain.py
==============
Agent team with a long-term brain:

  1. GraphRAG        — Neo4j graph + Pinecone vectors (hybrid search)
  2. Semantic Cache  — avoid repeat LLM calls for similar questions
  3. Checkpointing   — save/resume agent state across crashes

Flow:
  Query → Check semantic cache → Hybrid search (Graph + Vector) → LLM → Save checkpoint
"""

import os
import json
import hashlib
import time
from pathlib import Path

from neo4j import GraphDatabase
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from groq import Groq

# ── Config ────────────────────────────────────────────────────────────────────
NEO4J_URI      = os.environ.get("NEO4J_URI",       "bolt://localhost:7687")
NEO4J_USER     = os.environ.get("NEO4J_USER",      "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD",  "12345678")
PINECONE_KEY   = os.environ.get("PINECONE_API_KEY", "pcsk_4kAP3N_BsrKmCqcbSKwDt6WnMHu7TAxW89HL9wFoLesfjU6DZiHMtN9u4qcReU2WQsmr9S")
GROQ_KEY       = os.environ.get("GROQ_API_KEY",    "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")
INDEX_NAME     = "corporate-kb"
CHECKPOINT_DIR = Path("checkpoints")
CACHE_FILE     = Path("semantic_cache.json")

CHECKPOINT_DIR.mkdir(exist_ok=True)

# ── Clients ───────────────────────────────────────────────────────────────────
groq_client  = Groq(api_key=GROQ_KEY)
pc           = Pinecone(api_key=PINECONE_KEY)
pinecone_idx = pc.Index(INDEX_NAME)

embedder     = SentenceTransformer("all-MiniLM-L6-v2")


# ════════════════════════════════════════════════════════════════
# 1. SEMANTIC CACHE
# ════════════════════════════════════════════════════════════════

class SemanticCache:
    """
    Cache LLM responses by the semantic meaning of the question.
    If a nearly identical question was asked before, return cached answer.
    Threshold: cosine similarity > 0.92 = cache hit.
    """

    def __init__(self, cache_file: Path, threshold: float = 0.92):
        self.cache_file = cache_file
        self.threshold  = threshold
        self.cache      = self._load()

    def _load(self):
        if self.cache_file.exists():
            return json.loads(self.cache_file.read_text())
        return []

    def _save(self):
        self.cache_file.write_text(json.dumps(self.cache, indent=2))

    def _similarity(self, a, b):
        """Cosine similarity between two embedding lists."""
        dot   = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x ** 2 for x in a) ** 0.5
        mag_b = sum(x ** 2 for x in b) ** 0.5
        return dot / (mag_a * mag_b + 1e-9)

    def get(self, query: str):
        """Return cached answer if a similar query exists."""
        q_emb = embedder.encode(query).tolist()
        for entry in self.cache:
            sim = self._similarity(q_emb, entry["embedding"])
            if sim >= self.threshold:
                print(f"  [Cache HIT] similarity={sim:.3f} — skipping LLM call")
                return entry["answer"]
        return None

    def set(self, query: str, answer: str):
        """Store a new query-answer pair in the cache."""
        q_emb = embedder.encode(query).tolist()
        self.cache.append({"query": query, "embedding": q_emb, "answer": answer})
        self._save()
        print(f"  [Cache SET] stored answer for: '{query[:60]}...'")


# ════════════════════════════════════════════════════════════════
# 2. CHECKPOINTING
# ════════════════════════════════════════════════════════════════

class Checkpoint:
    """
    Save agent progress to disk after every step.
    If the agent crashes, resume from the last saved checkpoint.

    Checkpoint file format:
    {
      "task_id":      "research_acme_q3",
      "completed":    ["step_1_plan", "step_2_research"],
      "results":      {"step_1_plan": "...", "step_2_research": "..."},
      "last_updated": 1234567890
    }
    """

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.path    = CHECKPOINT_DIR / f"{task_id}.json"
        self.state   = self._load()

    def _load(self):
        if self.path.exists():
            state = json.loads(self.path.read_text())
            print(f"  [Checkpoint] Resuming task '{self.task_id}' "
                  f"— completed steps: {state['completed']}")
            return state
        return {"task_id": self.task_id, "completed": [], "results": {}, "last_updated": 0}

    def save(self, step_name: str, result: str):
        """Mark a step complete and save result."""
        self.state["completed"].append(step_name)
        self.state["results"][step_name] = result
        self.state["last_updated"] = int(time.time())
        self.path.write_text(json.dumps(self.state, indent=2))
        print(f"  [Checkpoint] Saved step '{step_name}'")

    def is_done(self, step_name: str) -> bool:
        """Check if a step was already completed (skip if so)."""
        return step_name in self.state["completed"]

    def get(self, step_name: str) -> str:
        """Retrieve a previously completed step's result."""
        return self.state["results"].get(step_name, "")

    def clear(self):
        """Delete checkpoint after full task completion."""
        if self.path.exists():
            self.path.unlink()
        print(f"  [Checkpoint] Task '{self.task_id}' complete — checkpoint cleared.")


# ════════════════════════════════════════════════════════════════
# 3. HYBRID SEARCH (Graph + Vector)
# ════════════════════════════════════════════════════════════════

def graph_search(query: str, limit: int = 3) -> str:
    mock_data = [
        "Sarah Chen --[CEO_OF]--> Acme Corp",
        "Acme Corp --[ACQUIRED]--> Beta Inc",
        "Acme Corp --[HEADQUARTERED]--> San Francisco",
        "Sarah Chen --[FOUNDED]--> Acme Corp",
        "Acme Corp revenue 2024 was $2.3 billion",
        "Beta Inc specializes in AI-powered logistics software",
    ]
    keywords = [w for w in query.lower().split() if len(w) > 3]
    results = [r for r in mock_data if any(k in r.lower() for k in keywords)]
    return "Graph knowledge:\n" + "\n".join(results) if results else ""

def vector_search(query: str, top_k: int = 3) -> str:
    """
    Search Pinecone for semantically similar document chunks.
    Returns the most relevant text passages.
    """
    q_emb = embedder.encode(query).tolist()
    hits   = pinecone_idx.query(vector=q_emb, top_k=top_k, include_metadata=True)

    passages = [match["metadata"]["text"] for match in hits["matches"]]
    return "Relevant passages:\n" + "\n---\n".join(passages) if passages else ""


def hybrid_search(query: str) -> str:
    """
    Combine graph search (relationships) + vector search (passages).
    Returns merged context for the LLM.
    """
    graph   = graph_search(query)
    vectors = vector_search(query)
    parts   = [p for p in [graph, vectors] if p]
    return "\n\n".join(parts) if parts else "No relevant knowledge found."


# ════════════════════════════════════════════════════════════════
# 4. AGENT CALL WITH CACHE
# ════════════════════════════════════════════════════════════════

def agent_call(role: str, system: str, query: str, context: str, cache: SemanticCache) -> str:
    """
    Call the LLM for an agent, using semantic cache to skip repeat calls.
    """
    full_query = f"{role}: {query}"

    # Check cache first
    cached = cache.get(full_query)
    if cached:
        return cached

    # Call LLM with knowledge base context injected
    messages = [
        {"role": "system",    "content": system},
        {"role": "user",      "content": (
            f"Question: {query}\n\n"
            f"Use this verified knowledge from our corporate knowledge base:\n"
            f"{context}\n\n"
            f"Answer based strictly on the knowledge provided. "
            f"If the knowledge base doesn't cover something, say so explicitly."
        )},
    ]

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=600,
    )

    answer = response.choices[0].message.content.strip()

    # Store in cache
    cache.set(full_query, answer)
    return answer


# ════════════════════════════════════════════════════════════════
# 5. CONTENT NEWSROOM WITH LONG-TERM BRAIN
# ════════════════════════════════════════════════════════════════

def run_newsroom_with_brain(topic: str, task_id: str = None):
    """
    Run the 3-agent newsroom (Editor → Researcher → Writer)
    with hybrid search, semantic caching, and checkpointing.
    """

    task_id    = task_id or hashlib.md5(topic.encode()).hexdigest()[:8]
    cache      = SemanticCache(CACHE_FILE)
    checkpoint = Checkpoint(task_id)

    print(f"\n{'='*60}")
    print(f"  NEWSROOM WITH LONG-TERM BRAIN")
    print(f"  Topic   : {topic}")
    print(f"  Task ID : {task_id}")
    print(f"{'='*60}\n")

    # ── Step 1: Editor plans ──────────────────────────────────────
    STEP_PLAN = "step_1_editor_plan"

    if checkpoint.is_done(STEP_PLAN):
        print(f"[SKIP] {STEP_PLAN} already done.")
        plan = checkpoint.get(STEP_PLAN)
    else:
        print("[EDITOR] Searching knowledge base...")
        context = hybrid_search(topic)

        print("[EDITOR] Planning article...")
        plan = agent_call(
            role    = "Editor",
            system  = (
                "You are a senior news editor. "
                "Using verified corporate knowledge provided, "
                "create a concise article plan: angle, audience, tone, 3 key questions."
            ),
            query   = f"Create an editorial plan for: {topic}",
            context = context,
            cache   = cache,
        )
        checkpoint.save(STEP_PLAN, plan)
        print(f"\nEDITOR PLAN:\n{plan}\n")

    # ── Step 2: Researcher gathers data ───────────────────────────
    STEP_RESEARCH = "step_2_researcher"

    if checkpoint.is_done(STEP_RESEARCH):
        print(f"[SKIP] {STEP_RESEARCH} already done.")
        research = checkpoint.get(STEP_RESEARCH)
    else:
        print("[RESEARCHER] Searching knowledge base...")
        context = hybrid_search(f"{topic} facts statistics background")

        print("[RESEARCHER] Gathering research...")
        research = agent_call(
            role    = "Researcher",
            system  = (
                "You are a meticulous news researcher. "
                "Using verified corporate knowledge, gather key facts, "
                "statistics, and background. Flag anything not in the knowledge base."
            ),
            query   = f"Research key facts for: {topic}\n\nEditorial plan:\n{plan}",
            context = context,
            cache   = cache,
        )
        checkpoint.save(STEP_RESEARCH, research)
        print(f"\nRESEARCH:\n{research}\n")

    # ── Step 3: Writer drafts article ─────────────────────────────
    STEP_WRITE = "step_3_writer"

    if checkpoint.is_done(STEP_WRITE):
        print(f"[SKIP] {STEP_WRITE} already done.")
        article = checkpoint.get(STEP_WRITE)
    else:
        print("[WRITER] Drafting article...")
        article = agent_call(
            role    = "Writer",
            system  = (
                "You are a news writer. "
                "Write a clear, engaging article using the research provided. "
                "Do not add facts not supported by the knowledge base."
            ),
            query   = (
                f"Write a news article about: {topic}\n\n"
                f"Plan:\n{plan}\n\n"
                f"Research:\n{research}"
            ),
            context = research,
            cache   = cache,
        )
        checkpoint.save(STEP_WRITE, article)

    # ── Done ───────────────────────────────────────────────────────
    checkpoint.clear()

    print("\n" + "="*60)
    print("  FINAL ARTICLE")
    print("="*60)
    print(article)

    with open(f"article_{task_id}.txt", "w", encoding="utf-8") as f:
        f.write(f"Topic: {topic}\n\n{article}")
    print(f"\nSaved to article_{task_id}.txt")

    return article


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    topic = input("Enter topic: ").strip() or "Acme Corp Q3 2024 earnings"
    run_newsroom_with_brain(topic)
