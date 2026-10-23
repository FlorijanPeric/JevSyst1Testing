# Email router: session notes (2026-10-06)

## What we built
- **`data/colleagues.csv`**: the 5 people emails get sorted to. Fields with more than one item use `|` between items.
  - Aleksander Tošič: Bitcoin & blockchain, his own processes
  - Niki Hrovatin: 3D printing & woodworking
  - Domen Vake: machine learning & AI
  - Jernej Vičič: CEO, research papers, partnerships, anything random
  - Tilen Jesenko: websites, domains, hosting
  - Email addresses are still `@example.com` placeholders. Replace them with the real ones.
- **`data/questions.json`**: extra system-one questions asked about every email (`type`, `needs_reply`, `urgency`). Edit this file to add more.
- **`run.sh`**: one script that runs everything (see below).

## How to run
```
./run.sh                   load colleagues, then type emails one by one (empty line quits)
./run.sh "email text"      sort one email
./run.sh email.txt         sort an email saved in a file
./run.sh --details "..."   also show match scores and timings
./run.sh load              embed only new/changed people from the CSV
./run.sh load --force      re-embed everyone
./run.sh forget            clear remembered routing decisions
./run.sh test              accuracy on data/test_emails.csv (106 labeled emails)
./run.sh rank [file.csv]   rank emails worst-first; worst 10 go to postman/ (default file: data/my_inputs.csv = every email you typed)
./run.sh inbox [file.csv]  priority inbox: most urgent first, spam last; saves data/inbox_sorted.csv
```

## Setup (`.env`, gitignored, holds the real keys)
- **Embeddings**: `EMBEDDING_API_URL=http://hivecore.famnit.upr.si:6666/api/embed`, model `bge-m3` (1024 dimensions).
- **System one**: `CHAT_MODEL=systemone/Qwen3-4B-Instruct-2507-Q4_K_M`, called at `HIVECORE_BASE_URL/systemone`.
- **Qdrant Cloud**: `QDRANT_URL` must be the base cluster URL only, with no `/dashboard...` on the end.
- **Collections**:
  - `COLLEAGUES_COLLECTION=JevTry`: ONE point per colleague, holding all their profile lines as a multivector (MaxSim, Cosine, 1024 dimensions). A `profile_hash` in the payload means only changed people get re-embedded, and people removed from the CSV are deleted.
  - `routed_emails`: cache of past routing decisions.
  - `QDRANT_COLLECTION=support_kb`: knowledge base used by `ingest.py` and `query.py`.

## People and tests (2026-10-07)
- `data/colleagues.csv` = 5 real people + 10 fictional ones (role ends in "(fictional)"). The real-only version is `data/colleagues.real.csv`.
- `data/test_emails.csv` = 106 emails labeled with the expected person, including 6 that should go to `manual_triage`.
- Accuracy: system one alone 62%, search alone ~75%, combined 77%, combined + more specific profiles + a `manual_triage` profile in the CSV (current) 92% (98/106, manual 5/6).
- The specifics were written after looking at the failures, so write fresh test emails to get an honest number.
- Decision: P(system one) + 5 x (search score - best score); manual only if system one gives it >= 0.7. Settings: `--search-weight`, `--manual-threshold`, `--candidates` (5).

## Confidence logging (2026-10-09)
- `./run.sh "..."` prints `Confidence:` = system one's `confidence` for its own pick (and its probability for the final pick if the search overruled it).
- `data/my_inputs.csv` logs every typed email: time, email, got, path, search_score, system_one_choice, confidence, final_pick_prob.
- `./run.sh test` and `./run.sh rank` show confidence per email; `rank` also writes it to `data/worst_queries.csv` and the Postman folder names.
- Typical confidence is only ~30-40% even when right (6 options), so compare emails with each other rather than reading it as an absolute number.

## Raising confidence (in progress, stopped 2026-10-09)
Experiment `experiments/conf_experiment.py` (run from the project folder) on the 106 test emails:

| Variant | Accuracy | Avg confidence (right / wrong) |
|---|---|---|
| current descriptions, 5 candidates | 96/106 | 42% (42% / 40%) |
| current descriptions, 3 candidates | 96/106 | 53% (53% / 53%) |
| clean descriptions, 5 candidates | not run yet | |
| clean descriptions, 3 candidates | not run yet | |

- Fewer candidates gives higher confidence at the same accuracy, but confidence then no longer separates right from wrong answers (53% / 53%).
- "Clean descriptions" = role + responsibilities only, without the "Name, role handles emails like:" prefix repeated on every line, plus a clearer instruction. Not measured yet.
- Qdrant Cloud timed out once during the run. Before rerunning, add a retry around `shortlist` and `timeout=30` on the client, and compute `shortlist` only on a cache miss (the script currently uses `setdefault`, which runs the search every time).
- Not changed in the real code yet: `route_email.py` still uses 5 candidates and the old descriptions.

## System one features (2026-10-23)
- `data/questions.json` now also asks `spam` (noul) and `language` (sl / en / other).
- Spam filter: spam >= 0.8 (`--spam-threshold`) means the email is not forwarded ("nobody, spam"). Tested: an iPhone scam scored 0.99, normal emails about 0.6.
- `query.py` grades its own answer with system one (`answers_question`, `uses_context_only`).
- Recipient options are now short: role + duties, best-matching duty first, no names. NOT MEASURED YET: Qdrant was down. Run `./run.sh test` and compare with 96/106 (old descriptions, 5 candidates). If it's worse, revert `recipient_question` in route_email.py.
- Qdrant Cloud free clusters get suspended after about a week without use. Resume the cluster in the Qdrant Cloud console if you get "Connection reset by peer".

## Long benchmark on server 125 (started 2026-10-23)
- Server: `ssh 125` (172.16.160.125 via Gateway; run `ssh-add ~/.ssh/keys/Gateway` first). Project in `~/RagAdd`, own `.venv`, `.env` copied by hand.
- `experiments/run_all.sh`: (1) `generate_emails.py` makes ~330 labeled emails with DeepSeek, falling back to Qwen3-30B, into `experiments/data/generated_emails.csv`; (2) `bench.py`: 8 chunkings x 3 aggregations (search only), then the best 3 + current x old/new descriptions x 3/5 candidates with system one, in rounds that drop losers; weights and thresholds swept offline with 2-fold CV.
- Smart exits: sqlite cache (rerun = resume), `MAX_HOURS` budget (default 6), Ctrl-C/kill writes the report first, circuit breaker if the API is down, model fallback in the generator.
- Results: `experiments/results/report.md` (updated after every round). Log: `experiments/run.log`.
- First local result (106 emails, search only): `lines_clean` + `top2mean` 89.6% top-1 vs current `lines_prefixed` + `max` 84.9%.
- Run 1 (436 emails, 15 min, `experiments/results_run1/` on 125): search-only best `grouped` 81% top-1 vs current 73%. With system one, everything is 81.7-83.3%. Old descriptions 83.0% with manual 53%; new (live) 81.7% with manual 25%, so going back to the old descriptions is advised. Bug: phase 2 tested `single` 3 times.
- Run 2 (started 10:48): selection by MRR, distinct chunkings: grouped, window3, lines_role + current. Results in `experiments/results/report.md`, log `experiments/run2.log`.

## Applied on this PC (2026-10-23), from benchmark runs 1-2
- Back to the OLD recipient descriptions (name + role + 2 best-matching lines): same accuracy, much better at catching emails that belong to nobody.
- Scoring by `grouped` chunking: JevTry points now have two named multivectors, `grouped` (duties block + examples block, averaged = the score) and `lines` (for the descriptions).
- Fast path 0.50 / 0.07 (was 0.65 / 0.10): 23% of emails skip system one at 98% accuracy.
- Test on 106: 94/106, manual 6/6 (was 2/6), fast path 14/14 correct.
- Server 125 still runs the old code, on purpose.
- Git: https://github.com/FlorijanPeric/JevSyst1Testing (`data/colleagues.csv` and `.env` are not in git).

## Gotchas we hit
- Scripts read `.env`, not `.env.example`.
- A collection's vector size must match the embedding model's output (1024). The one first made in the UI was 1536, so it was deleted and recreated.
- The colleagues collection used to have one point per line (19 Domens). It is now one point per person (multivector).
- The system-one response is nested: `answers.<question>.<type>`. `classify()` flattens it to `{question: value}`.
- A system-one `score` is the expected position in the list (0 = first label), not a value between 0 and 1.
- The old model name `systemone/Qwen3.5-4B-Q4_K_M` timed out: no worker on hivecore is running it.
- Hivecore rate-limits requests (`429`), so don't send many emails in quick succession.
- Don't paste lines starting with `#`, or the terminal prompt text, into zsh.

## Ideas for next steps
1. Connect a real inbox (IMAP), then auto-forward or label new mail. **Recommended first.**
2. Rules on top of the answers (spam is dropped, urgent mail is also copied to Jernej, and so on).
3. Suggested replies using `query.py` and a knowledge base in `data/`.
4. A `fix` command so corrections teach the cache.
5. A small web page, reading text from attachments and PDFs, and a statistics dashboard.
