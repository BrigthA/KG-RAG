# GraphMind — Knowledge Graph RAG (KG-RAG)

A Streamlit-based knowledge graph question-answering application using **LangChain, Neo4j AuraDB, OpenAI, LangSmith, and DeepEval**. It extracts entities and relationships from PDF/TXT files or a public HTTPS webpage, persists them in Neo4j, and answers questions using retrieved source evidence.

> **Status:** Starter/reference implementation. Validate quality, security, and reliability before production deployment.

## 1. Architecture

```text
PDF / TXT upload ──> PDF/Text loader ─────┐
                                         ├─> Text chunking (1200 chars; 150 overlap)
Public HTTPS URL ──> Requests + Trafilatura┘           |
                                                     v
                                       LangChain LLMGraphTransformer
                                       (entity and relation extraction)
                                                     |
                                       Schema validation / allowlists
                                                     |
                                            Neo4j AuraDB
                                       Entity + Chunk + RELATED
                                                     |
User question -> Input guardrail -> Question entity extraction
                                                     |
                                         Neo4j Cypher retrieval
                                                     |
                                      LLM answer with evidence IDs
                                                     |
                                         Output guardrail
                                                     |
                                    Streamlit chatbot + citations

LangSmith: traces instrumented pipeline stages
DeepEval: on-demand faithfulness and answer-relevancy evaluation
```

## 2. Technology stack

| Layer | Technology | Responsibility |
|---|---|---|
| Interface | Streamlit | Sidebar navigation, file/URL ingestion, chatbot, graph explorer, evaluation |
| Orchestration | LangChain | Document objects, splitting, LLM graph transformation |
| LLM | OpenAI via `langchain-openai` | Extract entities, identify query entities, generate grounded answers |
| Graph store | Neo4j AuraDB + `neo4j` Python driver | Persist and query graph with parameterized Cypher |
| Web extraction | Requests + Trafilatura | Fetch one HTTPS HTML page and extract readable text |
| PDF extraction | pypdf | Extract text and page numbers from text-based PDFs |
| Guardrails | Python regex checks + Pydantic + graph schema allowlists | Validate questions, graph structures and responses |
| Observability | LangSmith `@traceable` | Trace ingestion, retrieval, answer generation and checks |
| Evaluation | DeepEval | On-demand faithfulness and answer relevancy |

## 3. Project files

```text
kg-rag/
├── kg_rag_app.py
├── requirements_kg_rag.txt
├── .env                 # local secrets; do not commit
├── .gitignore
└── README.md
```

## 4. Prerequisites and configuration

1. Install Python 3.11 or 3.12.
2. Create a **Neo4j AuraDB** instance at https://console.neo4j.io/ and copy its URI, username and password.
3. Obtain an OpenAI API key; obtain a LangSmith API key if you want tracing.
4. Create and activate a virtual environment:

   **Windows PowerShell:**
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS/Linux:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

5. Install dependencies:

   ```bash
   python -m pip install -r requirements_kg_rag.txt
   ```

6. Create `.env` alongside `kg_rag_app.py`:

   ```dotenv
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_MODEL=gpt-4.1-mini

   NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_neo4j_password
   NEO4J_DATABASE=neo4j

   LANGSMITH_API_KEY=your_langsmith_api_key
   LANGSMITH_TRACING=true
   LANGSMITH_PROJECT=GraphMind-KG-RAG
   ```

7. Add `.env`, `.venv/`, `__pycache__/` and `.streamlit/secrets.toml` to `.gitignore`.
8. Launch:

   ```bash
   python -m streamlit run kg_rag_app.py
   ```

The application verifies Neo4j connectivity at startup and creates uniqueness constraints for `Entity.id` and `Chunk.id`.

## 5. Implementation: document and URL ingestion

### Step 1 — Select the input

Open **Data ingestion** in the Streamlit sidebar. Upload one or more **PDF/TXT** files, enter a **public HTTPS webpage URL**, or provide both.

- PDF/TXT uploads are limited to **15 MB per file**.
- URL ingestion accepts **one public HTML webpage**, not a full website crawl.
- URL redirects are disabled; non-HTML content is rejected.
- The current URL loader does not automatically follow PDF hyperlinks or scrape linked pages.

### Step 2 — Load and clean content

- `load_documents()` extracts text from PDF pages using `PdfReader`, or decodes TXT as UTF-8.
- `load_website()` validates the HTTPS URL, resolves the host, fetches HTML with Requests and extracts readable text with Trafilatura.
- Both loaders preserve the source; PDF pages retain actual page numbers. Website content is assigned page `1` as a placeholder.
- Scanned/image-only PDFs require an **OCR enhancement**; the current implementation does not perform OCR.

### Step 3 — Split and identify chunks

`RecursiveCharacterTextSplitter` splits extracted content with `chunk_size=1200` and `chunk_overlap=150`. Each chunk receives an ID based on a SHA-256 hash and its chunk index. This supports chunk-level provenance and repeated-content merging.

### Step 4 — Extract entities and relationships

`extract_graph()` calls LangChain's `LLMGraphTransformer` with an OpenAI chat model. The transformer is restricted to the configured entity types:

`Scheme`, `Budget`, `Department`, `Location`, `Crop`, `Organization`, `Amount`, `Year`, `Policy`.

Allowed relationships are:

`ALLOCATES`, `FUNDS`, `MANAGES`, `LOCATED_IN`, `SUPPORTS`, `APPLIES_TO`, `ANNOUNCED_IN`, `RELATED_TO`.

The prompt instructs the model to extract only facts explicitly present in the source. `validate_graph()` further filters nodes and edges to the allowlists. **LLM extraction can still make mistakes**; review critical facts before relying on them.

### Step 5 — Persist graph in Neo4j AuraDB

`store_graph()` uses parameterized Cypher and `MERGE` to store:

- `(:Entity {id, type})`: extracted entities.
- `(:Chunk {id, text, source, page})`: source text and provenance.
- `(entity)-[:MENTIONED_IN]->(chunk)`: source attribution.
- `(entity)-[:RELATED {kind, chunk_id}]->(entity)`: semantic relationship and supporting chunk ID.

**Important:** Relationship kinds are stored in the `kind` property of `RELATED`, rather than as separate Cypher relationship types. Entity IDs are globally merged by exact string; aliases and name collisions are not yet resolved.

## 6. Implementation: chatbot and graph query

### Step 6 — Validate user input

`input_guardrail()` rejects empty questions, questions longer than **500 characters**, and certain regex-detected prompt-injection, secret-extraction and destructive-command patterns. These are basic checks, not a complete prompt-injection defense.

### Step 7 — Extract search entities

`question_entities()` uses an LLM with structured output to identify up to five relevant names or domain concepts from the question.

### Step 8 — Retrieve graph evidence

`retrieve_graph()` executes a fixed, read-only, parameterized Cypher query. It matches entity IDs using case-insensitive equality/substring logic, then retrieves associated `Chunk` text and neighboring relationships. The current query limits entity matches to 15 and evidence chunks to 12.

This is **entity-based graph retrieval**, not hybrid vector + graph retrieval. Queries that omit matching entity names may return insufficient evidence.

### Step 9 — Generate an evidence-grounded answer

`answer_question()` sends retrieved text, source metadata, evidence IDs and graph relationships to the LLM. Pydantic's `Answer` model expects:

```python
class Answer(BaseModel):
    answer: str
    citations: list[str] = []
    sufficient_evidence: bool = False
```

The LLM is instructed to use only retrieved evidence and return supporting evidence IDs. If evidence is missing, the application returns an insufficient-evidence message.

### Step 10 — Validate output and show citations

`output_guardrail()` removes citation IDs not present in the retrieved evidence, blocks selected secret-like output patterns, and falls back if the answer lacks cited evidence. The chatbot shows the answer plus expandable source text and page information.

**Limitation:** A valid evidence ID is not proof that every factual claim is supported. Add claim-level verification for higher-stakes deployments.

## 7. Guardrail implementation reference

| Stage | Function | Current check |
|---|---|---|
| Input | `input_guardrail()` | Nonempty, max length, blocked regex patterns |
| Graph extraction | `validate_graph()` | Allowed node and relationship types |
| URL ingestion | `load_website()` | HTTPS, public-host DNS check, response size/type, no redirects |
| Answer generation | `answer_question()` | Evidence-only prompt, structured answer and citations |
| Output | `output_guardrail()` | Evidence ID validity, evidence sufficiency flag, secret-like patterns |

The current application uses **custom Python guardrails**, not the separate Guardrails AI library. URL DNS validation does not fully prevent DNS rebinding; production deployments need network-level egress restrictions.

## 8. LangSmith tracing

Set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT`. The code uses `@traceable` on ingestion, graph extraction, validation, storage, retrieval, guardrails, answer generation and DeepEval execution.

Use LangSmith to inspect pipeline steps, latency, exceptions and LLM calls. Traces may contain document content and user questions; configure access and retention accordingly.

## 9. DeepEval evaluation process

1. In **Chatbot**, ask a question and receive a cited response.
2. Select **Evaluation** from the sidebar.
3. Click **Run DeepEval**.
4. `run_deepeval()` builds an `LLMTestCase` using the question, generated answer and cited source chunks.
5. DeepEval runs **Answer Relevancy** and **Faithfulness** metrics, each with a configured threshold of `0.7`.
6. Review the score, pass/fail result and explanation displayed in the UI.

Evaluation is **on demand**, not a mandatory synchronous guardrail on every response. It may incur LLM API costs. A score is a quality signal, not a guarantee of correctness.

## 10. UI navigation

| Sidebar section | User workflow |
|---|---|
| **Chatbot** | Ask questions, view answers and inspect evidence |
| **Data ingestion** | Upload PDF/TXT, enter HTTPS URL and create graph |
| **Graph explorer** | Display up to 100 graph relationships in a table |
| **Evaluation** | Evaluate the latest chatbot answer using DeepEval |

The sidebar also offers **Clear conversation**. Conversation messages are held in Streamlit session state, not a persistent chat-history database.

## 11. Example

**Source sentence (illustrative):**

> The Tamil Nadu Agriculture Budget 2026–27 allocates ₹227.2 crore to farm mechanisation.

**Possible extracted graph:**

```text
(Tamil Nadu Agriculture Budget 2026–27:Budget)
       └── ALLOCATES ──> (₹227.2 crore:Amount)
       └── FUNDS ──────> (Farm Mechanisation:Scheme)
```

**Question:** How much funding was allocated to farm mechanisation?

**Expected behavior:** Retrieve the budget/source chunk, answer with the amount only if supported, and display the originating file or URL as evidence. Actual extracted nodes and relationships depend on the LLM output.

## 12. Neo4j graph inspection

Run in Neo4j Aura Query:

```cypher
MATCH (a:Entity)-[r:RELATED]->(b:Entity)
RETURN a, r, b
LIMIT 100;
```

Inspect sources for an entity:

```cypher
MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk)
RETURN e.id AS entity, e.type AS type,
       c.source AS source, c.page AS page, c.text AS text
LIMIT 25;
```

## 13. Troubleshooting

| Problem | Recommended check |
|---|---|
| `ModuleNotFoundError: neo4j` | Run `python -m pip install -r requirements_kg_rag.txt` in the same interpreter used by Streamlit |
| AuraDB connection fails | Check `NEO4J_URI`, credentials, instance status and network access |
| Website timeout | Enter a plain HTTPS URL (not Markdown), test in a browser and check connectivity; the code uses a 5-second connect timeout |
| URL rejected | Only public HTTPS HTML on port 443 is accepted; redirects are disabled |
| No PDF text extracted | PDF may be scanned; OCR is not implemented |
| No answer found | Check entity extraction and graph contents; exact/substring retrieval may miss synonyms |
| DeepEval fails | Check package installation, model access and OpenAI API key |
| No LangSmith traces | Verify tracing environment variables and project/key permissions |

## 14. Production roadmap

1. Add OCR for scanned PDFs and image-based government circulars.
2. Add approved-domain crawling, PDF-link discovery, URL deduplication and safer network egress controls.
3. Add entity normalization, aliases, confidence scoring and review of uncertain relationships.
4. Add Neo4j vector indexes, BM25/full-text search, graph traversal and reranking for **hybrid KG-RAG**.
5. Add claim-level groundedness checks, prompt-injection defenses, sensitive-data filtering and access control.
6. Build DeepEval regression datasets, CI evaluation thresholds and operational monitoring.
7. Add multi-user authentication, tenancy, ingestion jobs, retries and document lifecycle management.

## 15. Current limitations

- Web ingestion handles a single public HTML page per URL; it does not crawl a site or automatically extract linked PDFs.
- PDF ingestion extracts text but does not perform OCR.
- No embedding/vector database or hybrid semantic search is currently implemented.
- Graph extraction can introduce entity duplicates or unsupported relationships.
- Input/output checks are lightweight; DeepEval runs only when manually requested.
- No production authentication, tenant isolation or background ingestion workers are included.

---

**Implementation flow:** `Data Loader → Chunking → Entity Extraction → Relationship Mapping → Neo4j AuraDB → Input Guardrail → Graph Query → LLM Answer → Output Guardrail → Chatbot`, with LangSmith tracing throughout and optional DeepEval quality checks.
