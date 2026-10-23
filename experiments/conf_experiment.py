import csv, sys, types
sys.path.insert(0, "/Users/hentailover/Documents/RagAdd")
from qdrant_client import QdrantClient
import config
from clients import classify, embed_texts
from route_email import MANUAL, recipient_question, pick_recipient, shortlist

tests = list(csv.DictReader(open("data/test_emails.csv", encoding="utf-8")))
client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, timeout=30)
vectors = embed_texts([t["email"] for t in tests])
pts, _ = client.scroll(config.COLLEAGUES_COLLECTION, limit=100, with_payload=True)
profiles = {p.payload["person_id"]: p.payload for p in pts}

def clean_lines(pid):
    pl = profiles[pid]; prefix = f"{pl['name']}, {pl['role']}"
    return [t[len(prefix) + 2:] for t in pl["texts"] if t.startswith(prefix + ": ")]

def q_clean(people):
    crit = {}
    for p in people:
        if p["person_id"] == MANUAL: continue
        role = p["role"].replace(" (fictional)", "")
        crit[p["person_id"]] = f"{role}: " + "; ".join(clean_lines(p["person_id"]))
    crit[MANUAL] = "Spam, personal or off-topic email that none of these handle"
    return {"type": "choice", "instructions": "Which person is responsible for handling this email? Pick the one whose job matches its main topic.", "criteria": crit}

variants = {"clean k=5": (5, q_clean), "clean k=3": (3, q_clean)}
args = types.SimpleNamespace(search_weight=5.0, manual_threshold=0.7)
cache = {}
for name, (k, qfn) in variants.items():
    ok = 0; confs = []; ok_conf = []; bad_conf = []
    for t, v in zip(tests, vectors):
        key = (t["email"], k)
        if key not in cache:
            for attempt in range(4):
                try:
                    cache[key] = shortlist(client, v, k); break
                except Exception:
                    if attempt == 3: raise
                    import time; time.sleep(3)
        people = cache[key]
        ans = classify(t["email"], {"recipient": qfn(people)}, raw=True)["recipient"]
        final = pick_recipient(ans["probabilities"], people, args)
        good = final == t["expected"]; ok += good; confs.append(ans["confidence"])
        (ok_conf if good else bad_conf).append(ans["confidence"])
    avg = lambda xs: sum(xs) / len(xs) if xs else 0
    print(f"{name:<12} accuracy {ok}/{len(tests)}  avg conf {avg(confs):.0%}  (right {avg(ok_conf):.0%}, wrong {avg(bad_conf):.0%})", flush=True)
