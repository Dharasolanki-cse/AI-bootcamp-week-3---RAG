"""
knowledge_base.py
=================
Loads corporate knowledge into:
  - Neo4j  : entities and relationships (GraphRAG)
  - Pinecone: semantic vector embeddings

Run this ONCE to populate your knowledge base before running agents.
"""

from neo4j import GraphDatabase
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import os

NEO4J_URI      = os.environ.get("NEO4J_URI",      "bolt://127.0.0.1:7687")
NEO4J_USER     = os.environ.get("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD",  "12345678")
PINECONE_KEY   = os.environ.get("PINECONE_API_KEY", "pcsk_4kAP3N_BsrKmCqcbSKwDt6WnMHu7TAxW89HL9wFoLesfjU6DZiHMtN9u4qcReU2WQsmr9S")
INDEX_NAME     = "corporate-kb"

# ── Sample corporate knowledge ────────────────────────────────────────────────
# In production, load from PDFs, wikis, databases, etc.
CORPORATE_DOCS = [
    {
        "id": "doc1",
        "text": "Acme Corp was founded in 2005 by Sarah Chen and John Malik. "
                "Headquarters is in San Francisco. Revenue in 2024 was $2.3 billion.",
        "entities": [
            ("Acme Corp",  "Company"),
            ("Sarah Chen", "Person"),
            ("John Malik",  "Person"),
        ],
        "relations": [
            ("Sarah Chen", "FOUNDED",       "Acme Corp"),
            ("John Malik",  "FOUNDED",       "Acme Corp"),
            ("Acme Corp",  "HEADQUARTERED", "San Francisco"),
        ],
    },
    {
        "id": "doc2",
        "text": "Acme Corp acquired Beta Inc in 2022 for $400 million. "
                "Beta Inc specializes in AI-powered logistics software.",
        "entities": [
            ("Beta Inc", "Company"),
        ],
        "relations": [
            ("Acme Corp", "ACQUIRED", "Beta Inc"),
        ],
    },
    {
        "id": "doc3",
        "text": "Sarah Chen serves as CEO of Acme Corp. "
                "She previously worked at Google as VP of Engineering.",
        "entities": [
            ("Google", "Company"),
        ],
        "relations": [
            ("Sarah Chen", "CEO_OF",       "Acme Corp"),
            ("Sarah Chen", "WORKED_AT",    "Google"),
        ],
    },
    {
        "id": "doc4",
        "text": "Acme Corp's Q3 2024 earnings showed 18% YoY growth. "
                "The logistics division led growth at 34%. "
                "The board approved a $500M share buyback program.",
        "entities": [],
        "relations": [],
    },
]


# ── Neo4j: load graph data ────────────────────────────────────────────────────
def load_neo4j():
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        # Clear existing data
        session.run("MATCH (n) DETACH DELETE n")

        for doc in CORPORATE_DOCS:
            # Create entity nodes
            for name, label in doc["entities"]:
                query = f"MERGE (n:{label} {{name: $name}})"
                session.run(query, name=name)  # type: ignore[arg-type]

            # Create relationships
            for subj, rel, obj in doc["relations"]:
                session.run(
                    """
                    MERGE (a {name: $subj})
                    MERGE (b {name: $obj})
                    MERGE (a)-[r:REL {type: $rel}]->(b)
                    """,
                    subj=subj, obj=obj, rel=rel
                )

            # Store full document text as a node too
            session.run(
                "MERGE (d:Document {id: $id, text: $text})",
                id=doc["id"], text=doc["text"]
            )

    driver.close()
    print("Neo4j loaded successfully.")


# ── Pinecone: load vector embeddings ─────────────────────────────────────────
def load_pinecone():
    pc = Pinecone(api_key=PINECONE_KEY)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create index if it doesn't exist
    if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        import time
        print("Waiting for index to be ready...")
        time.sleep(30)  # wait 30 seconds for Pinecone to provision it
        print("Index ready.")

    index = pc.Index(INDEX_NAME)

    vectors = []
    for doc in CORPORATE_DOCS:
        embedding = model.encode(doc["text"]).tolist()
        vectors.append({
            "id":     doc["id"],
            "values": embedding,
            "metadata": {"text": doc["text"]},
        })

    index.upsert(vectors=vectors)
    print("Pinecone loaded successfully.")


if __name__ == "__main__":
    print("Loading knowledge base...")
    load_neo4j()
    load_pinecone()
    print("Done. Knowledge base ready.")
