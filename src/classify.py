import argparse
import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from src.models import ClassificationResult
from src.prompts import SYSTEM_PROMPT, format_ticket


load_dotenv()

client = OpenAI(
    base_url=os.environ["AZURE_OPENAI_BASE_URL"],
    api_key=os.environ["LLM_API_KEY"],
)

MODELS = {
    "nano": os.environ["GPT_NANO_DEPLOYMENT"],
    "gpt54": os.environ["GPT_54_DEPLOYMENT"],
}


def classify_ticket(ticket: pd.Series, model: str, max_retries: int = 5) -> tuple[ClassificationResult, dict]:
    """Classify one ticket and return the result and usage metrics."""
    for attempt in range(max_retries):
        try:
            start = time.perf_counter()
            response = client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": format_ticket(ticket)},
                ],
                text_format=ClassificationResult,
            )

            runtime = time.perf_counter() - start
            usage = response.usage
            reasoning_tokens = (
                usage.output_tokens_details.reasoning_tokens
                if usage.output_tokens_details else 0
            )

            metrics = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "visible_output_tokens": usage.output_tokens - reasoning_tokens,
                "total_tokens": usage.total_tokens,
                "runtime_seconds": runtime,
                "retries": attempt,
            }
            return response.output_parsed, metrics

        except RateLimitError:
            if attempt == max_retries - 1:
                raise

            wait_time = 2 ** attempt
            print(f"Rate limit exceeded. Retrying in {wait_time}s...")
            time.sleep(wait_time)


def classify_tickets(tickets: pd.DataFrame, model: str) -> tuple[pd.DataFrame, dict]:
    """Classify all tickets and collect run statistics."""
    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_reasoning_tokens = 0
    total_visible_output_tokens = 0
    total_tokens = 0
    total_api_runtime = 0.0
    total_retries = 0

    batch_start = time.perf_counter()

    for _, ticket in tickets.iterrows():
        result, metrics = classify_ticket(ticket, model)

        total_input_tokens += metrics["input_tokens"]
        total_output_tokens += metrics["output_tokens"]
        total_reasoning_tokens += metrics["reasoning_tokens"]
        total_visible_output_tokens += metrics["visible_output_tokens"]
        total_tokens += metrics["total_tokens"]
        total_api_runtime += metrics["runtime_seconds"]
        total_retries += metrics["retries"]

        results.append({
            "ticket_id": ticket["ticket_id"],
            "categorie": result.categorie.value,
            "prioriteit": ticket["prioriteit"],
            "te_valideren": int(result.te_valideren),
            "security_flag": int(result.security_flag),
            "reden": result.reden,
        })

        print(
            f"{ticket['ticket_id']}: {result.categorie.value} "
            f"({metrics['runtime_seconds']:.2f}s, {metrics['total_tokens']} tokens)"
        )

    batch_runtime = time.perf_counter() - batch_start

    summary = {
        "model": model,
        "api_calls": len(tickets),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "reasoning_tokens": total_reasoning_tokens,
        "visible_output_tokens": total_visible_output_tokens,
        "total_tokens": total_tokens,
        "api_runtime_seconds": total_api_runtime,
        "wall_clock_runtime_seconds": batch_runtime,
        "average_runtime_per_ticket": batch_runtime / len(tickets),
        "retries": total_retries,
    }
    return pd.DataFrame(results), summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="nano")
    return parser.parse_args()


if __name__ == "__main__":
    from src.data import load_tickets

    args = parse_args()
    model = MODELS[args.model]
    tickets = load_tickets()
    predictions, metrics = classify_tickets(tickets, model)

    predictions_path = f"outputs/voorspellingen_{args.model}.csv"
    metrics_path = f"outputs/run_metrics_{args.model}.json"

    predictions.to_csv(predictions_path, index=False)
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"\nSaved {len(predictions)} predictions to {predictions_path}")
    print(f"Model: {metrics['model']}")
    print(f"API calls: {metrics['api_calls']}")
    print(f"Input tokens: {metrics['input_tokens']}")
    print(f"Output tokens: {metrics['output_tokens']}")
    print(f"Reasoning tokens: {metrics['reasoning_tokens']}")
    print(f"Visible output tokens: {metrics['visible_output_tokens']}")
    print(f"Total tokens: {metrics['total_tokens']}")
    print(f"Runtime: {metrics['wall_clock_runtime_seconds']:.2f}s")
    print(
        f"Average runtime/ticket: {metrics['average_runtime_per_ticket']:.2f}s")
    print(f"Retries: {metrics['retries']}")
