# CrewAI RAG — Document Q&A with Multi-Agent Crew

A **Retrieval-Augmented Generation (RAG)** project built with [CrewAI](https://crewai.com). Index your own PDFs, then ask questions and get answers grounded in your documents — powered by a three-agent crew that retrieves, analyzes, and responds.

## What this project does

There are two phases:

### 1. Indexing (offline)

`vector_store.py` reads PDFs from a folder you configure, splits them into chunks, converts each chunk to a vector embedding (via OpenAI), and stores them in a local **Chroma** vector database.

```
PDFs in DOCUMENTS_FOLDER/
    → split into ~500-character chunks
    → embed with OpenAI
    → store in chroma_db/
```

### 2. Querying (runtime)

A CrewAI crew runs three tasks in sequence:

| Step | Agent | Role |
|------|--------|------|
| 1 | **Retriever** | Searches the vector store for the most relevant chunks |
| 2 | **Analyst** | Cleans up results — removes duplicates, filters noise |
| 3 | **Supervisor** | Writes the final answer from the refined context |

```
Your question
    → Retriever (vector_search tool) → raw chunks
    → Analyst → structured summary
    → Supervisor → final answer
```

## What is RAG?

**RAG = Retrieval-Augmented Generation**

A plain LLM only knows what it was trained on. It cannot read your PDFs unless you paste them in — and long documents do not fit in a single prompt.

RAG adds a retrieval step before generation:

1. Take the user's question
2. Search your documents for relevant passages
3. Pass those passages to the LLM as context
4. Generate an answer grounded in that context

This improves **accuracy**, allows **private/custom documents**, and makes answers **traceable** to source files.

## What is a vector database?

A **vector database** (here: [Chroma](https://www.trychroma.com/)) stores text as **embeddings** — lists of numbers that capture meaning. Similar ideas end up close together in vector space, so a question about *"AI in cybersecurity"* can match a passage about threat detection even when the exact words differ.

Chroma stores thousands of chunk vectors and quickly returns the **top-k most similar** matches. The `vector_search` tool uses `k=5` by default.

The database is stored locally at `chroma_db/` and is rebuilt from your PDFs — it is not committed to git.

## Requirements

- Python **3.10 – 3.13**
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- An **OpenAI API key**
- PDF files to index

## Setup

### 1. Clone and enter the project

```bash
git clone git@github.com:SebasSotoA/crew-ai-rag.git
cd crew-ai-rag
```

### 2. Install dependencies

Using uv (recommended):

```bash
uv sync
```

Or with CrewAI CLI:

```bash
pip install uv
crewai install
```

### 3. Configure environment

Copy the example env file and edit it:

```bash
cp .env.example .env
```

Set these values in `.env`:

```env
MODEL=gpt-4o-mini
OPENAI_API_KEY=your-openai-api-key-here
DOCUMENTS_FOLDER=C:/path/to/your/pdf-folder
```

`DOCUMENTS_FOLDER` is the folder containing the PDFs you want to query. Every `**/*.pdf` file inside it will be indexed.

### 4. Build the vector store

**Required before running the crew.** Use the same virtual environment you will use to run the app:

```bash
uv run python -m rag_crew.vector_store
```

Or, with the project venv activated:

```bash
python -m rag_crew.vector_store
```

This clears any previous index and rebuilds from the PDFs in `DOCUMENTS_FOLDER`.

### 5. Run the crew

```bash
crewai run
```

Or directly:

```bash
uv run python -m rag_crew.main
```

Edit the query in `src/rag_crew/main.py` or pass inputs via the CrewAI API.

## Interactive console (recommended)

A menu-driven console app lets you explore the index, rebuild it, and ask questions without editing code:

```bash
uv run rag_console
```

Or:

```bash
uv run python -m rag_crew.console
```

| Menu option | What it does |
|-------------|--------------|
| **1) Explore** | List indexed PDFs, chunk counts, optional search preview |
| **2) Rebuild** | Delete `chroma_db/` and re-index all PDFs in `DOCUMENTS_FOLDER` |
| **3) Ask** | Interactive Q&A loop — each question runs the full RAG crew |
| **4) Exit** | Quit |

**Typical workflow:**

1. Copy PDFs into the folder set in `DOCUMENTS_FOLDER` (`.env`)
2. Start the console: `uv run rag_console`
3. Choose **Rebuild** (first time, or when PDFs change)
4. Choose **Ask** and type questions
5. Use **Explore** anytime to verify what is indexed

Asking questions does **not** re-index — only **Rebuild** clears and recreates the database.

## Inspecting the vector store

List indexed files and chunk counts:

```bash
uv run python -m rag_crew.vector_store inspect
```

Preview search results with source file and page:

```bash
uv run python -m rag_crew.vector_store inspect "your search query"
```

## Project structure

```
.
├── src/rag_crew/
│   ├── main.py              # Entry point — runs the crew
│   ├── console.py           # Interactive menu (explore / rebuild / ask)
│   ├── crew.py              # Agent and task definitions
│   ├── vector_store.py      # PDF indexing + inspect command
│   ├── config/
│   │   ├── agents.yaml      # Agent roles and goals
│   │   └── tasks.yaml       # Task descriptions
│   └── tools/
│       └── vector_search_tool.py  # Chroma similarity search
├── chroma_db/               # Local vector index (git-ignored)
├── .env                     # Your secrets (git-ignored)
├── .env.example             # Template for .env
├── pyproject.toml           # Dependencies
└── uv.lock                  # Locked dependency versions
```

## Important notes

- **Always rebuild the vector store** after adding, removing, or moving PDFs.
- **Use one virtual environment** — build the index and run the crew from the same `.venv` (inside this project folder). Mixing environments can cause ChromaDB version mismatches.
- **Never commit `.env`** — it contains your API key. Only `.env.example` is tracked.
- **`chroma_db/` is local** — each developer rebuilds their own index from their PDFs.

## Customization

- **Agents** — edit `src/rag_crew/config/agents.yaml`
- **Tasks** — edit `src/rag_crew/config/tasks.yaml`
- **Query** — edit `src/rag_crew/main.py`
- **Chunk size / search results** — edit `vector_store.py` and `vector_search_tool.py`

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Wrong or unrelated answers | Rebuild the vector store; run `inspect` to verify indexed files |
| `OPENAI_API_KEY is required` | Create `.env` from `.env.example` and set your key |
| ChromaDB crash / sqlite panic | Rebuild index using this project's `.venv`, not a different environment |
| `crewai run` dependency errors | Run `uv sync` from this directory |

## License

MIT (or your chosen license)
