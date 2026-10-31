"""STEP 7 - grounded generation in isolation (no databases needed; needs OPENAI_API_KEY).

Feeds fixed chunks to the LLM and checks it (a) answers with citations and (b) refuses
when the sources don't contain the answer.

    python scripts/07_answer.py
"""
from _common import DATA
from rag.chunking import chunk_documents
from rag.config import Settings
from rag.generation import REFUSAL, AnswerGenerator, OpenAIChat
from rag.io import read_documents

s = Settings.from_env()
sales_chunks = [c for c in chunk_documents(read_documents(DATA)) if "sales" in c.allowed_teams][:5]
generator = AnswerGenerator(OpenAIChat(s.openai_api_key, s.llm_model))

for question in ("What is the refund limit for invoice INV-12345?", "Who won the 2018 World Cup?"):
    a = generator.answer(question, sales_chunks)
    print(f"Q: {question}\nA: {a.text}\n   grounded={a.grounded}")
    for c in a.citations:
        print(f"   [{c.index}] {c.title} -> {c.source_url}")
    print()
