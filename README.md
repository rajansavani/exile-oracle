# exile-oracle

A question-answering tool for Path of Exile (PoE 1), backed by the PoE Wiki.

Live: https://exile-oracle-yfgngtxg4q-uc.a.run.app

Ask it about skills, unique items, keystones, ascendancies, mechanics, or patch notes. Answers are grounded in the wiki and cite the pages they came from.

## Try it!

- What does the Acrobatics keystone do?
- What ascendancy is best for a minion build?
- What are some minion skills I can use early game?
- What does Headhunter do?

## Known limitations

Short one-word queries like "shock" or "block" retrieve poorly because many unrelated pages mention those words. Phrasing the question ("how does shock stack against bosses?") usually helps.

The corpus is the first section of each wiki page, not the full page. Deep mechanics questions may get partial answers. A fuller corpus is a future item.

PoE 1 only. PoE 2 support would be a separate corpus and is not built.

## How it works

This is a retrieval-augmented generation (RAG) system. The app looks up relevant wiki pages for your question, then asks a language model to answer using those pages as its source material. The LLM isn't allowed to make things up from memory, it has to stay grounded in the retrieved text and cite the wiki pages referenced.

The pipeline runs in stages. Ingest pulls text from about 4,900 PoE Wiki pages via the MediaWiki API. Chunk splits each page into ~500-token overlapping chunks. Embed sends every chunk to OpenAI's `text-embedding-3-small` and caches the resulting vectors to disk (about 7,300 total). Index upserts those vectors into a Pinecone serverless index with the chunk text stored as metadata. Retrieve embeds your question with the same model and asks Pinecone for the 5 most similar chunks. Generate hands those chunks to `gpt-4o-mini` along with a system prompt that tells it to only use the provided context and cite sources by title.

The app itself is a FastAPI service with a minimal HTML front-end, containerized with Docker and deployed to Google Cloud Run. Pytest runs in GitHub Actions on every push.

## Stack

Python, FastAPI, Pydantic, OpenAI (embeddings + chat), Pinecone (serverless vector store), Docker, Google Cloud Run, GitHub Actions.

## Running locally

You need Python 3.12, an OpenAI API key, and a Pinecone API key.

    git clone https://github.com/rajansavani/exile-oracle.git
    cd exile-oracle
    python -m venv venv
    venv\\Scripts\\Activate           # Windows
    source venv/bin/activate         # macOS/Linux
    pip install -r requirements.txt

    cp .env.example .env
    # edit .env and paste in your OPENAI_API_KEY and PINECONE_API_KEY

    # build the corpus (1-2 hours and about $0.50 in embedding cost, one time)
    python -m src.ingest
    python -m src.chunk
    python -m src.embed
    python -m src.index

    # run the server
    uvicorn src.main:app --reload --port 8000

Open http://localhost:8000 and ask a question.

## Project layout

    src/
      config.py     settings loaded from environment variables
      ingest.py     scrape wiki pages via MediaWiki API
      chunk.py      token-based chunking with overlap
      embed.py      OpenAI embeddings with on-disk caching
      index.py      Pinecone index creation and upsert
      retrieve.py   semantic search against Pinecone
      generate.py   grounded answer generation with gpt-4o-mini
      main.py       FastAPI app with /ask, /health, and static UI
    tests/                      smoke tests
    .github/workflows/ci.yml    pytest and Docker build on every push

## Attribution

Content is sourced from the Path of Exile Wiki (https://www.poewiki.net/), available under CC BY-NC-SA 3.0. This project is not affiliated with Grinding Gear Games.
