# Corpus

The development and evaluation corpus is a set of public arXiv papers about
retrieval-augmented generation. The PDFs are **not committed** (`data/` is
git-ignored); this file records exactly which versions were used so the
corpus can be rebuilt.

Each paper is pinned to a specific arXiv version (`v1`, `v2`, ...), because
a newer version can change the text and silently invalidate evaluation labels.

| File in `data/raw/arxiv/` | arXiv ID | Title |
|---|---|---|
| `2005.11401v4_rag_knowledge_intensive_nlp.pdf` | 2005.11401v4 | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks |
| `2004.04906v3_dense_passage_retrieval.pdf` | 2004.04906v3 | Dense Passage Retrieval for Open-Domain Question Answering |
| `2312.10997v5_rag_for_llms_survey.pdf` | 2312.10997v5 | Retrieval-Augmented Generation for Large Language Models: A Survey |
| `2310.11511v1_self_rag.pdf` | 2310.11511v1 | Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection |
| `2401.15884v3_corrective_rag.pdf` | 2401.15884v3 | Corrective Retrieval Augmented Generation |
| `2212.10496v1_hyde_zero_shot_dense_retrieval.pdf` | 2212.10496v1 | Precise Zero-Shot Dense Retrieval without Relevance Labels |
| `2307.03172v3_lost_in_the_middle.pdf` | 2307.03172v3 | Lost in the Middle: How Language Models Use Long Contexts |
| `2309.15217v2_ragas_evaluation.pdf` | 2309.15217v2 | Ragas: Automated Evaluation of Retrieval Augmented Generation |
| `2404.16130v2_graph_rag_local_to_global.pdf` | 2404.16130v2 | From Local to Global: A Graph RAG Approach to Query-Focused Summarization |

Titles were checked against the arXiv API (`export.arxiv.org/api/query`).

## Rebuilding the corpus

Download each file from `https://arxiv.org/pdf/<arXiv ID>` and save it under
the file name in the table, for example in PowerShell:

```powershell
New-Item -ItemType Directory -Force data\raw\arxiv
Invoke-WebRequest https://arxiv.org/pdf/2005.11401v4 -OutFile data\raw\arxiv\2005.11401v4_rag_knowledge_intensive_nlp.pdf
```

## Licensing

Papers are downloaded for local, personal research use and are not
redistributed. Each paper's license is listed on its arXiv abstract page.
