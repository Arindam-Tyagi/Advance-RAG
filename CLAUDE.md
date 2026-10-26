# CLAUDE.md — working rules for this repository

An advanced, modular RAG system built incrementally. Each milestone is tested
and evaluated before the next one starts.

## Approval protocol (mandatory)

- Before each major milestone, report: objective, verified current state,
  proposed file changes, technology choices, alternatives, risks/costs, and a
  short list of decisions needing approval. **Wait for approval.**
- Ask before choosing: LLM provider/model, embedding model, vector store,
  document formats, chunking/BM25/fusion, reranker, router design, Graph RAG,
  Text-to-SQL, LangChain/LangGraph components, LangSmith, eval methodology,
  logging/monitoring/deployment, or anything that costs money or sends user
  documents to a third party.
- Ask before adding any dependency; explain its purpose first.
- Use established conventions for trivial details; document them instead of asking.
- Never invent file paths, APIs, env vars, model capabilities, test results or
  metrics. If something cannot be verified, say it is unknown.
- Never overwrite or delete existing work, commit, push, create remotes or
  expose services without explicit approval.

## Environment

- Python 3.12, uv-managed virtual environment at `.venv/`.
- Package `rag_app` uses the `src/` layout: `src/rag_app/`.
- Add dependencies with `uv add <pkg>` (dev-only: `uv add --dev <pkg>`) so
  `pyproject.toml` and `uv.lock` stay in sync. Do not `pip install` into `.venv`.
- Run commands through `uv run ...`.
- Configuration comes from environment variables / `.env` (template: `.env.example`).

## Architecture

- Keep layers separate: ingestion, chunking, embeddings, retrieval,
  reranking, generation, routing, evaluation, API, UI.
- Define small interfaces so retrieval strategies can be swapped without
  rewriting callers.
- Graph RAG and Text-to-SQL, if adopted, stay isolated from the core vector
  pipeline and are optional.
- The Streamlit UI (later) only calls the backend; no retrieval or generation
  logic in the UI.
- Create a module when its milestone is implemented, not before. No circular
  imports, no speculative abstractions.

## Coding standards

- Readable Python with type hints and docstrings on public functions/classes.
- Explicit error handling; never swallow exceptions silently.
- No hardcoded paths, credentials or unexplained magic numbers; tunables go in config.
- Small focused modules; no duplicated logic.

## AI engineering standards

- Establish a baseline before any optimization; never assume a more complex
  retrieval strategy is better.
- Compare strategies on the same evaluation dataset and report retrieval
  metrics (Recall@k, Precision@k, MRR, nDCG), answer quality (correctness,
  faithfulness, citation validity), latency and token cost.
- Keep actual measurements separate from illustrative examples. Never fabricate results.
- Preserve original document text and useful metadata through the pipeline.
- Respect embedding model token limits.
- Citations must reference chunks that were actually retrieved.
- Valid JSON is not proof of factual correctness.
- Agent/graph workflows use bounded retries and explicit failure paths.

## Security

- Never hardcode or log secrets.
- Do not commit private documents, generated vector stores, databases or model weights.
- Treat retrieved documents as untrusted input (prompt injection).
- Consider authorization and data leakage when exposing sources or diagnostics.
- Never execute LLM-generated SQL without validation, read-only access and restrictions.

## Workflow per change

1. Explain the objective and design decisions.
2. List the files that will change.
3. Implement one coherent feature at a time.
4. Run relevant tests and report the actual output.
5. Explain what changed, how to run it, and its limitations.

Never claim a test passed or a metric improved without running it and observing the result.
