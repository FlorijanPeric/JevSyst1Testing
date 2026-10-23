"""Generate a bigger labeled test set with another model (DeepSeek on hivecore).

The generator only sees each person's role, never their duty lines or example emails,
so the emails are not copies of the profiles. Resumable: rerun to fill up missing ones.

Output: experiments/data/generated_emails.csv (expected,email,source)
"""
import argparse
import csv
import random
import time
from pathlib import Path

from openai import OpenAI

from common import EXP_DIR, ROOT, Breaker, GracefulStop, Stop, log
import config

OUT = EXP_DIR / "data" / "generated_emails.csv"
MANUAL = "manual_triage"
TONES = ["polite", "annoyed", "very brief", "rambling", "urgent", "casual with typos", "formal", "confused"]
SENDERS = ["a coworker", "a student", "a customer", "a manager", "an external partner", "a new employee", "a researcher"]
OFF_TOPIC = ["spam or a scam", "a personal question unrelated to work", "an ad for a product", "a message clearly sent to the wrong company",
             "someone selling a private item", "a newsletter", "a question about something no tech company handles (e.g. a dentist, a restaurant)"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-person", type=int, default=20)
    parser.add_argument("--manual", type=int, default=30, help="Emails that belong to nobody")
    parser.add_argument("--models", default="DeepSeek-V4-Flash,hf.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:UD-Q4_K_XL",
                        help="Generator models in order of preference; a model that fails is skipped for a while")
    args = parser.parse_args()

    stopper = GracefulStop()
    breaker = Breaker(stopper)
    llm = OpenAI(base_url=config.HIVECORE_BASE_URL, api_key=config.HIVECORE_API_KEY, timeout=60, max_retries=0)
    models = args.models.split(",")
    benched: dict[str, float] = {}  # model -> time until which it is skipped

    with (ROOT / "data" / "colleagues.csv").open(encoding="utf-8") as f:
        roles = {r["id"]: r["role"].replace(" (fictional)", "") for r in csv.DictReader(f) if r["id"] != MANUAL}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, int] = {}
    if OUT.exists():
        with OUT.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done[r["expected"]] = done.get(r["expected"], 0) + 1
    new_file = not OUT.exists()
    out = OUT.open("a", encoding="utf-8", newline="")
    writer = csv.writer(out)
    if new_file:
        writer.writerow(["expected", "email", "source"])
        out.flush()

    todo = [(pid, f"someone whose job is: {role}") for pid, role in roles.items()
            for _ in range(max(0, args.per_person - done.get(pid, 0)))]
    todo += [(MANUAL, None) for _ in range(max(0, args.manual - done.get(MANUAL, 0)))]
    random.shuffle(todo)
    log(f"{sum(done.values())} already generated, {len(todo)} to go")

    def ask(prompt: str) -> tuple[str, str]:
        """Try models in order; one that fails is benched for 10 minutes so the others carry on."""
        errors = []
        for model in models:
            if benched.get(model, 0) > time.time():
                continue
            try:
                r = llm.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                                temperature=1.1, max_tokens=350)
                text = (r.choices[0].message.content or "").strip()
                if text:
                    return text, model
                errors.append(f"{model}: empty")
            except Exception as e:
                errors.append(f"{model}: {str(e)[:80]}")
            benched[model] = time.time() + 600
            log(f"  {model} failed, skipping it for 10 min")
        if all(benched.get(m, 0) > time.time() for m in models):
            benched.clear()  # everything failed: let the breaker wait, then try all again
        raise RuntimeError("; ".join(errors) or "all models benched")

    try:
        for i, (pid, target) in enumerate(todo, 1):
            if stopper.requested:
                raise Stop("stop requested")
            tone, lang = random.choice(TONES), random.choice(["English", "English", "Slovenian"])
            n = random.choice([1, 2, 3, 4])
            if pid == MANUAL:
                what = f"Write an email that is {random.choice(OFF_TOPIC)}."
            else:
                what = (f"Write an email from {random.choice(SENDERS)} that only {target} would be the right person "
                        f"to handle. Pick a concrete, specific situation. Do not mention anyone's job title.")
            prompt = (f"{what} Tone: {tone}. Language: {lang}. About {n} sentences. "
                      "Output only the email body, no subject, no greeting placeholders like [Name].")
            text, model = breaker.call(ask, prompt)
            writer.writerow([pid, text, model])
            out.flush()
            if i % 20 == 0:
                log(f"generated {i}/{len(todo)}")
            time.sleep(0.3)
    except Stop as e:
        log(f"Stopped early: {e}")
    finally:
        out.close()
    log(f"Done. File: {OUT}")


if __name__ == "__main__":
    main()
