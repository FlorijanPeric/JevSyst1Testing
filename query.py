import argparse
import time

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

import config
from clients import chat_complete, classify, embed_texts

# Same schema as the systemone addon example, reused to route retrieval.
TRIAGE_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department handles this?",
        "criteria": {
            "returns": "Returns and exchanges",
            "shipping": "Delivery issues",
            "billing": "Charges and refunds",
        },
    },
    "refund": {
        "type": "noul",
        "instructions": "Is the customer asking for a refund?",
    },
    "severity": {
        "type": "score",
        "instructions": "How severe is the problem?",
        "criteria": [
            "Minor",
            "Moderate: wrong item",
            "Major: safety or financial loss",
        ],
    },
}


# Asked about the generated answer; each is a 0-1 probability of "yes".
ANSWER_CHECK = {
    "answers_question": {"type": "noul", "instructions": "Does the answer actually answer the customer's ticket?"},
    "uses_context_only": {"type": "noul", "instructions": "Is everything in the answer supported by the context?"},
}


def search(client: QdrantClient, vector: list[float], top_k: int, department: str | None):
    query_filter = None
    if department:
        query_filter = Filter(must=[FieldCondition(key="department", match=MatchValue(value=department))])
    return client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=vector,
        query_filter=query_filter,
        limit=top_k,
    ).points


def timed_search(client, vector, top_k, department):
    start = time.perf_counter()
    hits = search(client, vector, top_k, department)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return hits, elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a ticket, retrieve routed context, and answer it.")
    parser.add_argument("text", help="Customer ticket text")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--no-route", action="store_true", help="Skip department filtering (search whole collection)")
    parser.add_argument("--benchmark", action="store_true", help="Compare filtered vs unfiltered search latency")
    args = parser.parse_args()

    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)

    triage = classify(args.text, TRIAGE_QUESTIONS)
    print(f"Triage: {triage}")
    department = None if args.no_route else triage.get("department")

    vector = embed_texts([args.text])[0]

    if args.benchmark:
        routed_hits, routed_ms = timed_search(client, vector, args.top_k, department)
        full_hits, full_ms = timed_search(client, vector, args.top_k, None)
        print(f"Filtered search (department={department}): {routed_ms:.2f} ms, {len(routed_hits)} hits")
        print(f"Unfiltered search (whole collection):       {full_ms:.2f} ms, {len(full_hits)} hits")
        hits = routed_hits
    else:
        hits = search(client, vector, args.top_k, department)

    context = "\n\n".join(f"[{h.payload['source']}] {h.payload['text']}" for h in hits)
    system_prompt = (
        "You are a customer support agent. Answer using only the provided context. "
        f"Ticket triage: department={triage.get('department')}, "
        f"refund_requested={triage.get('refund')}, severity={triage.get('severity')}."
    )
    user_prompt = f"Context:\n{context}\n\nCustomer ticket:\n{args.text}"

    answer = chat_complete(system_prompt, user_prompt)
    print("\n--- Answer ---")
    print(answer)

    # Let system one grade the generated answer, so bad answers can be caught before they are sent.
    check = classify(f"Context:\n{context}\n\nCustomer ticket:\n{args.text}\n\nAnswer:\n{answer}", ANSWER_CHECK)
    print("\n--- Answer check (system one) ---")
    for qid, value in check.items():
        print(f"  {qid:<18} {value:.2f}  {'ok' if value >= 0.5 else 'CHECK THIS'}")


if __name__ == "__main__":
    main()
