"""Legixo Grounded Q&A — document-grounded RAG API.

Package layout:
    config       typed settings loaded from .env
    clients      Gemini / Groq / Pinecone client + embedding factories
    schemas      FastAPI request/response models
    chunking     Markdown-aware, header-preserving chunking
    vectorstore  Pinecone index lifecycle, upsert, query
    ingestion    corpus -> chunks -> embeddings -> Pinecone pipeline
    retrieval    embed + search + score-filter + dedupe
    llm          grading / query rewriting / answer generation / citation guard
    graph        the LangGraph StateGraph wiring the above into a workflow
    main         FastAPI application
"""
