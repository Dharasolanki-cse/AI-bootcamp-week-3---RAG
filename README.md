# Agent Long-Term Brain

## What this covers
- GraphRAG with Neo4j (entity + relationship graph)
- Semantic vector search with Pinecone
- Hybrid Graph + Vector search to reduce hallucinations
- Semantic caching (skip repeat LLM calls)
- Checkpointing (resume after crash)

## How each piece works

### GraphRAG (Neo4j)
Stores WHO relates to WHAT:
  Sarah Chen --[CEO_OF]--> Acme Corp
  Acme Corp  --[ACQUIRED]--> Beta Inc
When an agent asks "Who runs Acme Corp?", Neo4j traverses
the graph and returns the relationship — not just a text match.

### Vector Search (Pinecone)
Stores the MEANING of documents as embeddings.
When an agent asks a question, it finds semantically similar
passages even if the exact words don't match.

### Hybrid Search
Combines both: graph gives RELATIONSHIPS, vectors give PASSAGES.
Together they give the LLM grounded, structured context
which dramatically reduces hallucinations.

### Semantic Cache
Stores (question_embedding → LLM_answer) pairs.
If similarity > 0.92 with a past question, returns cached answer.
Saves Groq API calls and speeds up repeated queries.

### Checkpointing
Saves agent state to checkpoints/<task_id>.json after every step.
If the process crashes mid-task, re-running it will:
  - detect the checkpoint
  - skip already-completed steps
  - resume from where it left off

## Setup

### 1. Neo4j (free local option)
Download Neo4j Desktop: https://neo4j.com/download/
Or run with Docker:
  docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j

### 2. Pinecone (free tier)
Sign up at: https://www.pinecone.io/
Get your API key from the dashboard.

### 3. Install dependencies
  pip install -r requirements.txt

### 4. Set environment variables (Windows)
  set GROQ_API_KEY=gsk_xxxxxxxx
  set PINECONE_API_KEY=xxxxxxxx
  set NEO4J_URI=bolt://localhost:7687
  set NEO4J_USER=neo4j
  set NEO4J_PASSWORD=password

### 5. Load the knowledge base (run once)
  python knowledge_base.py

### 6. Run the agents
  python agent_brain.py

## Testing checkpoint recovery
1. Add `raise Exception("Simulated crash")` after step 1 in agent_brain.py
2. Run — it will crash after the editor step
3. Remove the exception and run again
4. It will skip step 1 and resume from step 2

## Files
  knowledge_base.py  — loads Neo4j + Pinecone with corporate data
  agent_brain.py     — 3-agent newsroom with brain, cache, checkpoints
  requirements.txt   — Python dependencies
  checkpoints/       — auto-created, stores agent progress
  semantic_cache.json — auto-created, stores cached LLM answers
