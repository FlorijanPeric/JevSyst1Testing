"""End-to-end check of the running service (service.py): routing, speed, fix -> profile learning, ok, summary.

    .venv/bin/python experiments/api_selftest.py http://127.0.0.1:8091
Meant for a temporary collection and temporary log files (see run_round3.sh).
"""
import sys
import time
from pathlib import Path

import requests

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8090"


def route(email: str, learn: bool = True) -> dict:
    t = time.perf_counter()
    d = requests.post(f"{BASE}/route", json={"email": email, "learn": learn}, timeout=120).json()
    total = (time.perf_counter() - t) * 1000
    print(f"  -> {d['forward_to']:<14} {d['decided_by']:<22} server {d['ms']:>5} ms, round trip {total:>5.0f} ms   {email[:50]}")
    return d


print("health:", requests.get(f"{BASE}/health", timeout=30).json())

print("\nrouting via the API (the second SSL email is a repeat: embedding cache + routing cache):")
for email in ["The SSL certificate on our website expired",
              "The SSL certificate on our website expired",
              "Can you book a hotel in Vienna for the conference?",
              "Win a free iPhone now, click this link!",
              "Hi, how was your holiday at the seaside?",
              "Re: invoice\n\nPlease pay the attached supplier invoice by Friday.\n\nOn Mon, 3 Nov 2026, Eva wrote:\n> Which invoice?"]:
    route(email)

print("\nfix -> the correction is learned into the profile (fixed to someone it did NOT go to, so it must change):")
email, related = "The iPad version of our app cuts off the bottom menu", "On tablets the app hides part of the navigation bar"
before, related_before = route(email)["forward_to"], route(related, learn=False)["forward_to"]  # not cached, or the cache would answer later
target = "tilen" if "tilen" not in (before, related_before) else "matej"
print(" ", requests.post(f"{BASE}/fix", json={"email": email, "person_id": target}, timeout=300).json()["message"])
print("  related, not identical email afterwards:")
after = route(related)["forward_to"]
print(f"  learning: {'PASS' if after == target else 'FAIL'} (related email went to {related_before} before, "
      f"{after} after, fixed to {target})")

print("\nok:", requests.post(f"{BASE}/ok", json={"email": "Can you book a hotel in Vienna for the conference?"},
                             timeout=30).json()["message"])
print("unknown person is rejected:", requests.post(f"{BASE}/fix", json={"email": "x", "person_id": "nobody_here"},
                                                   timeout=30).status_code)
print("\nsummary:\n" + requests.get(f"{BASE}/summary", params={"days": 1}, timeout=30).text)
