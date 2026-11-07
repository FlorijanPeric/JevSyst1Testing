# What this router can be used for

## In one paragraph

You give it a piece of text (usually an email) and a list of people with what each of them handles. It says
who should get it, or that nobody should (spam, personal, wrong address). It does this in three steps:
a **search** that compares the text with each person's profile, **memory** of similar emails that were already
sorted, and **system one** (a small model on hivecore) for the unclear ones. Every correction you make is
remembered, so it gets better and cheaper with use.

## What it can do today (measured, rounds 3-8)

| | Result |
|---|---|
| Right person on a fresh system | ~88-90% (106 handwritten + 330 generated test emails) |
| Clear matches, decided without system one | ~25% of emails, ~98% right |
| After using it for a while (with feedback) | more than half of emails skip system one; memory decisions 95-100% right |
| One correction, then a differently worded email about the same thing | 22% → 99% right, other emails not affected |
| Languages | Slovenian and English (also mixed) |
| Speed | 10-400 ms for clear or remembered emails, ~1-8 s when system one is asked |
| System one down | still routes, on search only |
| Extra answers per email | spam probability, urgency, language, or any yes/no or category question you add |

## Where it fits

The pattern is always the same: **many messages come into one place, and someone has to decide who handles each one.**

### Email
- **Shared addresses** (info@, office@, a department or project address). It forwards each email to the right
  colleague instead of one person reading and forwarding everything.
- **University or faculty offices**: student questions go to the student office, the right professor, IT or
  international office; theses, grants and papers to the right people. Works in Slovenian.
- **Priority inbox** (`./run.sh inbox`): sorts by urgency, with spam last. Useful for one busy person, not only for routing.
- **Spam and "not for us" filter**: the three nobody profiles plus the spam question. Wrong-recipient and
  personal emails are kept apart from real requests.

### Helpdesk and tickets
- **IT or support tickets**: assign to a person or team (network, printers, accounts, web, security).
  The decision log and `summary` show how load is spread per person.
- **Customer contact forms**: billing, technical, sales, partnerships, complaints.
- **Bug reports**: to the developer or component owner (web, mobile app, backend, ML).

### Beyond email (same code, different text)
- **Chat messages** (Slack or Teams channel for "who can help with...?").
- **Documents and requests**: invoices to finance, contracts to legal, CVs to HR, grant calls to the project lead.
- **Phone or meeting notes**: route action items to the person responsible.
- **Finding the expert**: "who in the company knows about X?" is the same search, used as a question
  instead of forwarding.

## Why it works well for this

- **No training phase**: a CSV with people and what they handle is enough to start. New person = new row.
- **Learns from normal work**: `fix` when wrong, `ok` when right. No retraining, no data science.
- **Cheap**: most emails never reach a model once memory has grown; the model is small (4B) and local (hivecore).
- **Explainable**: every decision says how it was made (clear match, past emails like it, system one,
  spam filter) and with what confidence. All in `data/decisions.jsonl`.
- **Data stays in-house**: hivecore and Qdrant on our own servers; nothing goes to an outside AI service.
- **Easy to connect**: a web API (`./run.sh serve`: `/route`, `/fix`, `/ok`, `/summary`) or the command line.

## Limits to know

- **~10% wrong at the start**, so it should first **suggest**, not forward on its own.
- **One recipient per email**; it does not split an email that is for two people.
- **Reads the text only**: no attachments or PDFs yet.
- **Profiles must be honest**: if two people's descriptions overlap, it will mix them up. Corrections fix this over
  time, but a clear CSV helps from day one.
- **Strange or brand-new topics** go to manual triage until someone corrects a few of them.
- Tested on our own test sets (generated and handwritten); real mail will be messier. The real test is
  real emails with real corrections.

## How to roll it out

1. **Shadow mode** (1-2 weeks): it routes every incoming email but only logs. Someone uses `ok` and `fix`.
   This builds the memory and gives real accuracy numbers (`./run.sh test data/real_labels.csv`).
2. **Suggest mode**: it adds a label or note ("suggested: Tilen, clear match"), and a person forwards with one click.
3. **Auto mode for the safe paths**: forward automatically when the decision came from a clear match or from
   memory (95-100% right); keep system one decisions as suggestions until their real accuracy is known.
4. **Re-tune** once there are 100+ real labels (`./run.sh tune`), only if the honest estimate goes up.

## What it needs

- `data/colleagues.csv`: id, name, email, role, what they handle, a few example emails.
- hivecore access (system one + bge-m3 embeddings), and Qdrant (local docker on server 125).
- For real use: a connection to the mailbox (IMAP) or the ticket system. This is the main missing piece.

## Next steps that would add the most

1. **Mailbox connection** (IMAP) with shadow mode, then suggest mode.
2. **Attachments**: read text from PDFs and Word files.
3. **Re-ranking the top candidates** with `qwen3-reranker` on hivecore, to cut the remaining search misses.
4. **Suggested replies** for common questions, using the same memory of past emails.
