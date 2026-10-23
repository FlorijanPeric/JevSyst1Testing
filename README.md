# JevSyst1Testing

# Support-ticket RAG over Qdrant

Ingests department knowledge-base docs into Qdrant, then answers a customer
ticket by first classifying it with the `systemone` structured-answer addon
and using that classification to **filter** the vector search.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in HIVECORE_API_KEY and EMBEDDING_MODEL
```

**Qdrant:** if you already have the `JezikTech` container from Jezikovne
tehnologije running on `localhost:6333`, you don't need to start anything —
this app just adds a new collection called `support_kb` to that same
instance, alongside your existing `idioms_*` / `JezikovneTehnologije`
collections, without touching them. Check with:

```bash
docker ps --filter name=JezikTech
```

If it's not running, start it (`docker start JezikTech`) or, for a fresh
standalone instance instead, use `docker-compose up -d` (edit the ports in
`docker-compose.yml` first if 6333/6334 are already taken).

## Ingest

```bash
python ingest.py            # ingests data/returns.txt, shipping.txt, billing.txt
                             # each file's department tag = its filename
```

Add your own `.txt` files to `data/` (or point at another path/file), and use
`--department` to override the auto-detected tag for files that don't match
`returns` / `shipping` / `billing`.

## Query

```bash
python query.py "I ordered size 10 shoes but received size 8."
```

This:
1. Calls `/v1/systemone` with the department/refund/severity schema to triage the ticket.
2. Embeds the ticket text via `/v1/embeddings`.
3. Searches Qdrant **filtered to the classified department only**.
4. Sends the retrieved context + triage result to the chat model for the final answer.

Compare filtered vs. unfiltered retrieval:

```bash
python query.py "I ordered size 10 shoes but received size 8." --benchmark
```

## Email routing to colleagues

```bash
cp data/colleagues.example.json data/colleagues.json   # edit with real people (gitignored)
python ingest_colleagues.py
python ingest_colleagues.py data/colleagues.csv          # or a CSV; list columns are "|"-separated
python route_email.py "I was charged twice for order #4411"
python route_email.py path/to/email.txt
```

Each colleague is stored as several vectors (one per responsibility and
example email). Routing runs in two stages: Qdrant shortlists the top
`--candidates` people (grouped by `person_id`), then the systemone addon picks
one of them with a `choice` question, or `manual_triage` if nobody fits.
To make routing better, add more `example_emails`, ideally real past emails
each person handled.

### Faster than plain RAG

The LLM call is ~95% of the latency, so the router avoids it when it can:

| Path    | When                                                                 | Cost              |
|---------|----------------------------------------------------------------------|-------------------|
| `cache` | A past routed email has similarity >= `--cache-threshold` (0.95)     | embed + 1 search  |
| `fast`  | Top colleague scores >= `--min-score` (0.65) and beats #2 by `--min-margin` (0.10) | embed + search |
| `llm`   | Everything else (ambiguous emails)                                   | + addon call      |

Fast and LLM decisions are stored in the `routed_emails` collection, so
the cache grows as you use it (`--no-learn` disables this, `--forget` clears it).

```bash
python route_email.py --forget
python route_email.py --benchmark data/emails_sample.txt
```

The benchmark runs each email through both the smart router and an
always-LLM baseline. It prints which path each email took, the time per
email, the overall speedup, and how often both picked the same person.

**Tune the thresholds on your real data.** Similarity scores depend on the
embedding model. Run a single email and look at the printed scores, then
set `--min-score` / `--min-margin` so routine emails take the fast path and
ambiguous ones still go to the LLM. Raise the thresholds if the
"Agreement w/ LLM" count drops.

## Why classify-then-filter makes RAG faster

- **Smaller search space.** Qdrant's `query_filter` on the `department`
  payload field restricts the ANN search to one department's vectors instead
  of the whole collection. On a small demo collection the wall-clock
  difference is tiny, but it scales: with thousands of docs per department,
  searching 1/3 (or 1/N) of the index is proportionally faster and returns
  more relevant hits at the same `top_k`, so you don't need to over-fetch and
  re-rank.
- **Cheap classification, expensive generation.** The addon call is a small,
  constrained-schema completion (a handful of enum/bool/score fields) — much
  cheaper and faster than asking the LLM to reason over the full ticket in
  free text. Spend the expensive, unconstrained generation call only on
  writing the final answer from an already-narrowed context.
- **Early-exit potential.** The same triage fields can gate the pipeline
  before you even hit Qdrant — e.g. skip retrieval entirely for
  `severity: Minor` tickets that don't need policy lookup, or branch on
  `refund` to decide which knowledge base subset to search.
- **Note:** the assignment JSON used `"type": "noul"` for the refund
  question — this is almost certainly a typo for `"bool"`; `query.py` uses
  `"bool"`. If the addon actually expects the literal string `"noul"`,
  change `TRIAGE_QUESTIONS` in `query.py` to match.
