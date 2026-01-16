Here's your content converted into clean, well-structured Markdown format:
Markdown# MedWise RAG – Production-Grade Medical Retrieval-Augmented Generation System

## Overview

MedWise RAG is a **production-grade** medical question-answering system built using **Retrieval-Augmented Generation (RAG)**.  

The system is designed to:

- Answer medical questions **strictly** from authoritative PDF sources
- Minimize hallucinations
- Provide accurate citations
- Maintain low latency with **streaming responses**

Medical RAG systems are particularly challenging because incorrect or hallucinated answers can have serious consequences. This project focuses on building a **faithful, explainable, and observable** RAG pipeline — prioritizing correct retrieval, citation accuracy, and controlled inference latency over raw model output quality.

## System Architecture

The system follows a **modular, end-to-end architecture** designed for experimentation, observability, and production hardening.

### Data Ingestion & Indexing

- Medical PDFs are ingested, cleaned, and parsed with **page-level metadata**
- Documents are chunked using configurable chunk sizes and overlaps (enables systematic experimentation)
- Embeddings generated using lightweight **sentence-transformer** models optimized for CPU inference
- Stored in persistent **Chroma** vector database
- Vector store tracked via **DVC**

### Retrieval Layer

- Vector search with **Maximal Marginal Relevance (MMR)** → balances relevance and diversity
- Retrieval latency consistently **< 1 second** on CPU
- All retrieved chunks retain **page-level metadata** → critical for accurate citations

### RAG Orchestration

- Implemented using **LangChain (v1.x)**
- Strict prompt constraints to **prevent hallucinations**
- Enforces **source-grounded** answering
- Mandates **explicit page-number citations** for every factual claim

### Inference & Streaming

- Backend: **FastAPI** with fully asynchronous endpoints
- Answers streamed **token-by-token** using **Server-Sent Events (SSE)**
- Responsive user experience
- Tracks and exposes:  
  - Retrieval time  
  - Generation time  
  - End-to-end latency

### Frontend

- **Streamlit**-based chat interface
- ChatGPT-like experience
- Displays answers with **citations** and **latency metrics**

## Evaluation & Experiments

Fully **config-driven** experimentation framework  
→ Ran **10+ controlled experiments** varying:

- Chunk sizes
- Chunk overlaps
- Retrieval parameters
- Prompting strategies

### Metrics Tracked

- **Recall** (retrieval coverage)
- **Precision** (retrieval relevance)
- **Citation Accuracy** (faithful grounding)
- **Retrieval Latency**
- **Generation Latency**
- **End-to-End Latency**

All experiments tracked using **MLflow** (hosted on **Dagshub**) → full reproducibility & comparison

## Results

**Best performing configuration** achieved:

- Recall ≈ **0.40**
- Sub-second retrieval latency
- End-to-end latency ≈ **50–60 seconds** (CPU)
- **Strict citation correctness**

→ This configuration was **promoted to production**

## Observability & Monitoring

End-to-end production-grade observability:

- **MLflow + Dagshub** – experiment tracking & metrics
- **LangSmith** – RAG trace inspection & prompt debugging
- Structured Python logging (ingestion → inference)
- Health check endpoints for backend monitoring

All observability features are **environment-controlled** (config files + env variables)

## CI/CD & Hardening

Full production hardening:

- Continuous Integration via **GitHub Actions**
- Unit tests for: ingestion, retrieval, evaluation, API health
- **Dummy LLM** used in CI (no external dependencies)
- **Dockerized** backend + frontend
- Environment isolation (local / Docker / CI)
- **DVC**-based versioning for datasets & vectorstore
- Secrets/API keys managed **only** via environment variables (never committed)

## Tech Stack

- **Language**: Python 3.12
- **LLM Orchestration**: LangChain (v1.x)
- **LLM Inference**: Ollama (llama3.2)
- **Embeddings**: Sentence Transformers
- **Vector Store**: Chroma
- **Backend**: FastAPI (async + streaming)
- **Frontend**: Streamlit
- **Experiment Tracking**: MLflow + Dagshub
- **Tracing**: LangSmith
- **Testing**: Pytest
- **Data Versioning**: DVC
- **Containerization**: Docker, Docker Compose