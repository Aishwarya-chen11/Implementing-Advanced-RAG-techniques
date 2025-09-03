# Advanced RAG Pipeline — LlamaIndex + TruLens

**Sentence-Window Retrieval, Auto-Merging Retrieval, and the RAG-Triad (Context Relevance, Groundedness, Answer Relevance)**

> **TL;DR.** This project builds and evaluates a production-minded Retrieval-Augmented Generation (RAG) system that combines **Sentence-Window Retrieval (SWR)** for high-precision local context, **Auto-Merging Retrieval (AMR)** for coherent multi-chunk context, and the **RAG-Triad** of metrics (**Context Relevance, Groundedness, Answer Relevance**) to quantify quality and reduce hallucinations. A leaderboard logs quality, latency, tokens, and cost for side-by-side comparisons.

---

## Project Overview

Pure vector-similarity RAG often returns either too little context (misses key facts) or too much (noisy, redundant). This repo implements two complementary retrieval strategies:

* **Sentence-Window Retrieval (SWR):** retrieve the best sentence and *attach its neighbors* as context → **precision** around the relevant fact.
* **Auto-Merging Retrieval (AMR):** when multiple leaf chunks are retrieved, **merge** them into their shared parent section → **coherent**, non-fragmented context.

Quality is measured with the **RAG-Triad**:

* **Context Relevance (CR):** how relevant the retrieved context is to the query.
* **Groundedness (G):** how well the answer is supported by the retrieved context.
* **Answer Relevance (AR):** how directly the final answer addresses the query.

A dashboard/leaderboard (via TruLens) tracks **CR/G/AR**, **latency**, **tokens**, and **cost** across engines.

---

## Objectives

* **Maximize factual grounding** while **minimizing context bloat** by pairing SWR (precision) with AMR (coherence).
* Establish **quantitative guardrails** using CR/G/AR and operational metrics (latency, tokens, cost).
* Provide a **repeatable evaluation loop** to choose the best structure per document type.

---

## Repo / Notebook Map

* **`Advanced_RAG_Pipeline.ipynb`**  – end-to-end pipeline & comparisons. [Open Colab Notebook](https://github.com/Aishwarya-chen11/Implementing-Advanced-RAG-techniques/blob/main/Advanced_RAG_Pipeline.ipynb)
* **`RAG_Triad_of_metrics.ipynb`** – feedback functions and leaderboard wiring. [Open Colab Notebook](https://github.com/Aishwarya-chen11/Implementing-Advanced-RAG-techniques/blob/main/RAG_Triad_of_metrics.ipynb)
* **`Sentence_window_retrieval.ipynb`** – SWR parser, post-processors, window-size experiments. [Open Colab Notebook](https://github.com/Aishwarya-chen11/Implementing-Advanced-RAG-techniques/blob/main/Sentence_window_retrieval.ipynb)
* **`Auto-merging_Retrieval.ipynb`** – hierarchical nodes, AMR retriever, merge behavior. [Open Colab Notebook](https://github.com/Aishwarya-chen11/Implementing-Advanced-RAG-techniques/blob/main/Auto-merging_Retrieval.ipynb)

---

## System Design (high level)

1. **Ingestion & Indexing**

   * Parse documents; persist **VectorStoreIndex** to enable fair, repeatable runs.
2. **Three Retrieval Engines**

   * **Direct** (baseline dense retrieval)
   * **Sentence-Window Retrieval (SWR)**
   * **Auto-Merging Retrieval (AMR)**
3. **Generation**

   * LLM answers **only** from retrieved context.
4. **Evaluation & Observability**

   * RAG-Triad via TruLens; log **latency, tokens, total cost**; compare in a leaderboard.

---

## Implementation Details (code-backed)

### 1) Building & Implementing Advanced RAG (LlamaIndex)

* **Ingestion & consolidation:** `SimpleDirectoryReader` → consolidated `Document`s for uniform parsing.
* **LLM & embeddings:** e.g., `OpenAI(..., temperature=0.1)` + `"BAAI/bge-small-en-v1.5"` (or equivalent) via a shared `ServiceContext` to keep runs comparable.
* **Indexing & persistence:** `VectorStoreIndex` persisted (e.g., `./sentence_index`, `./merging_index`).
* **Comparable engines:** Direct, SWR, AMR are created with identical LLM/embedding settings for apples-to-apples evaluation.
* **Instrumentation:** `Tru()` + `TruLlama` recorder hooks (prebuilt helper) to capture feedback scores and operational metrics into a **Leaderboard**.

### 2) RAG-Triad of Metrics (Context Relevance, Groundedness, Answer Relevance)

**Feedback wiring (pseudocode reflecting the actual code):**

```python
provider = fOpenAI()  # LLM used to run evals

# Context Relevance: query ↔ retrieved contexts (aggregate across contexts)
context_selection = TruLlama.select_source_nodes().node.text
f_qs_relevance = (Feedback(provider.qs_relevance_with_cot_reasons, name="Context Relevance")
                  .on_input()           # pointer to user query
                  .on(context_selection) # pointers to retrieved contexts
                  .aggregate(np.mean))   # mean across all retrieved contexts

# Answer Relevance: query ↔ model output
f_qa_relevance = (Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
                  .on_input()
                  .on_output())

# Groundedness: model output ↔ retrieved contexts
grounded = Groundedness(groundedness_provider=provider)
f_groundedness = (Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
                  .on(context_selection)
                  .on_output()
                  .aggregate(grounded.grounded_statements_aggregator))
```

**Scalable vs Meaningful.** LLM/MLM/traditional metrics are highly **scalable**; **human/ground-truth** checks are more **meaningful** but lower throughput. This project leans on scalable LLM-based feedbacks and can layer targeted human checks later.

**Honest / Harmless / Helpful coverage.**

* **Honest:** *Answer Relevance, Context Relevance, Groundedness*, plus optional embedding distance, BLEU/ROUGE, and summarization quality.
* **Harmless (optional):** PII detection, toxicity, stereotyping, jailbreak checks.
* **Helpful (optional):** sentiment, language mismatch, conciseness, coherence.
  The Triad anchors “honesty,” while the framework is ready to add safety/helpfulness gates.

### 3) Sentence-Window Retrieval (SWR)

* **Parsing:** `SentenceWindowNodeParser.from_defaults(window_size=3)` → sentence-level nodes with **neighboring windows** in metadata.
* **Post-processing:**

  * `MetadataReplacementPostProcessor(target_metadata_key="window")` – replaces the single hit with its **full window** at answer time.
  * `SentenceTransformerRerank(model="BAAI/bge-reranker-base", top_n=2)` – sharpens top-k windows.
* **Querying:** `sentence_index.as_query_engine(similarity_top_k=6, node_postprocessors=[postproc, rerank])`.
* **Window-size experiments:** sweep **{1, 3, 5}** and observe Triad trade-offs: larger windows generally **raise CR** but can **increase tokens/latency**; smaller windows keep cost low but risk under-contexting.

### 4) Auto-Merging Retrieval (AMR)

* **Hierarchy:** `HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])` → parents & leaves stored in docstore.
* **Retriever:** wrap a base retriever with `AutoMergingRetriever(base_retriever, storage_context, verbose=True)` to **merge** sibling hits into their **shared parent** once a **threshold** is exceeded.
* **Query engine:** `RetrieverQueryEngine.from_args(automerging_retriever, node_postprocessors=[rerank])`.
* **Why it helps:** AMR reduces **fragmentation/duplication** and yields a **single, self-contained citation**, typically lifting **Groundedness** and **Context Relevance** at similar latency.

---

## Experiments

* Load **different question sets**; keep engines identical except for the retrieval strategy.
* **SWR:** try window sizes **1 → 3 → 5**, re-score with the Triad, and pick the **smallest window** that meets CR/G thresholds to control cost.
* **AMR:** iterate **hierarchy depth**, **parent/child sizes**, **merge thresholds**, and **top-k**, then re-score.
* Track all runs in the **Leaderboard** (records, avg latency, total tokens, total cost) to pick the best structure per document type.

---

## Results

### A) Engine comparison (20-ish records unless noted)

| Engine                    | Context Relevance | Groundedness | Answer Relevance | Avg Latency (s) | Cost (USD) | Tokens |
| ------------------------- | ----------------: | -----------: | ---------------: | --------------: | ---------: | -----: |
| **Direct Query Engine**   |          **0.26** |         0.80 |             0.93 |            2.20 |     \~0.03 |  \~19k |
| **Sentence-Window (SWR)** |          **0.34** |         0.88 |             0.93 |            2.25 |     \~0.02 |  \~11k |
| **Auto-Merging (AMR)**    |         **0.435** |     **1.00** |             0.94 |            2.25 |   0.000799 |      — |

**Takeaways**

* **AMR > SWR > Direct** on **Context Relevance** and **Groundedness**, while **Answer Relevance** stays consistently high.
* **Cost/token efficiency:** SWR and AMR improve grounding with **tighter or more coherent context** vs. naïve “dump top-k chunks.”

### B) Sentence-Window variants (leaderboard snapshots)

| SWR Engine             | Records | Groundedness | Context Relevance | Answer Relevance | Avg Latency (s) | Total Tokens | Total Cost |
| ---------------------- | ------: | -----------: | ----------------: | ---------------: | --------------: | -----------: | ---------: |
| **engine 1**           |      21 |         0.83 |              0.57 |             0.87 |            4.57 |      \~9.18k |   \~\$0.02 |
| **engine 3** *(pilot)* |       1 |         1.00 |              0.90 |             1.00 |            3.00 |          846 |      \~\$0 |
| **engine 5** *(pilot)* |       1 |         0.86 |              0.90 |             1.00 |            3.00 |        1.06k |      \~\$0 |

> *Pilot rows are single-record probes—promising but expand to ≥20 records before drawing firm conclusions.*

### C) App roll-up (single app configuration)

| app\_id    | Context Relevance | Groundedness | Answer Relevance | Latency (s) | Total Cost |
| ---------- | ----------------: | -----------: | ---------------: | ----------: | ---------: |
| **App\_1** |              0.56 |         0.86 |         0.918182 |    3.545455 |   0.000874 |

---

## Key Insights

* **Complementarity:** SWR excels at pinpoint facts; AMR shines on multi-paragraph explanations where scattered hits need to be returned as **one coherent passage**.
* **Grounding wins:** AMR delivered **G ≈ 1.0** and lifted **CR** without increasing latency, improving trust and debuggability.
* **Budget control:** SWR/AMR reduced **prompt tokens** versus dumping many top-k chunks, lowering cost with equal or better quality.

---

## Evaluate & Iterate (what to keep doing)

* **Systematically search the space:**

  * SWR → tune **window\_size**, **similarity\_top\_k**, **reranker top\_n**.
  * AMR → tune **hierarchy depth**, **chunk sizes**, **merge thresholds**.
* **Evaluate each variant with the RAG-Triad** and log latency/tokens/cost to optimize **quality-per-token**.
* **Choose by doc type:** e.g., contracts/invoices may favor **deeper hierarchies + AMR**; short policies often work with **small SWR windows**.
* **Guardrails:** enforce **Groundedness ≥ 0.9** and **Context Relevance ≥ 0.5** for auto-responses; otherwise return an **“insufficient context”** message or ask a clarifying question.
* **Safety/Helpfulness (optional gates):** add PII/toxicity/jailbreak checks and helpfulness signals (conciseness, coherence) for production.

---

## How to Reproduce (quick path)

1. **Index the corpus** with LlamaIndex and persist to disk.
2. **Create engines:** Direct, SWR (parser + metadata replacement + reranker), AMR (hierarchical nodes + auto-merge + reranker).
3. **Instrument with TruLens** using the three feedback functions above.
4. **Run the eval set** (same prompts across engines).
5. **Open the leaderboard** to compare **CR/G/AR**, latency, tokens, cost.
6. **Iterate** (SWR window sizes; AMR hierarchies) and re-score.

---

## Future Work

* **Cross-encoder reranking before AMR** to lift CR on long sections.
* **Query rewriting / multi-query fusion** to boost recall without bloating final context.
* **Layout-aware chunking** (headings/tables) so AMR merges into even cleaner parents.
* **Caching & prompt budgeting** per query; profile retrieval vs generation time; trim/merge intelligently.
* **Human spot-checks** on low-CR or low-G tails for high-stakes domains.

---

## License & Acknowledgements

* Built with **LlamaIndex** and **TruLens**; 
* Replace any API keys and model choices with your own before running.

---
