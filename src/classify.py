import json
import os
import time
from enum import Enum

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from openai import RateLimitError

from src.prompts import SYSTEM_PROMPT, format_ticket


class Category(str, Enum):
    HARDWARE = "Hardware"
    SOFTWARE = "Software"
    NETWORK = "Netwerk"
    ACCESS_ACCOUNTS = "Toegang & Accounts"
    PRINTER = "Printer"
    MOBILE_DEVICE = "Mobiel Toestel"
    SECURITY = "Security"
    APPLICATION_SUPPORT = "Applicatieondersteuning"
    OTHER = "Overig"


class ClassificationResult(BaseModel):
    categorie: Category
    te_valideren: bool
    security_flag: bool
    reden: str


load_dotenv()

client = OpenAI(
    base_url=os.environ["AZURE_OPENAI_BASE_URL"],
    api_key=os.environ["LLM_API_KEY"],
)

MODEL = os.environ["GPT_NANO_DEPLOYMENT"]


def classify_ticket(ticket: pd.Series, max_retries: int = 5) -> tuple[ClassificationResult, dict]:
    """Classify one ticket and return the result and usage metrics."""
    for attempt in range(max_retries):
        try:
            start = time.perf_counter()

            response = client.responses.parse(
                model=MODEL,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": format_ticket(ticket)},
                ],
                text_format=ClassificationResult,
            )

            runtime = time.perf_counter() - start
            usage = response.usage

            metrics = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens,
                "runtime_seconds": runtime,
                "retries": attempt,
            }

            return response.output_parsed, metrics
        except RateLimitError:
            if attempt == max_retries - 1:
                raise

            wait_time = 2 ** attempt
            print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)


def classify_tickets(tickets: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Classify all tickets and collect run statistics."""
    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0
    total_api_runtime = 0.0
    api_calls = 0
    total_retries = 0

    batch_start = time.perf_counter()

    for _, ticket in tickets.iterrows():
        result, metrics = classify_ticket(ticket)

        api_calls += 1
        total_input_tokens += metrics["input_tokens"]
        total_output_tokens += metrics["output_tokens"]
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
        "model": MODEL,
        "api_calls": api_calls,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "api_runtime_seconds": total_api_runtime,
        "wall_clock_runtime_seconds": batch_runtime,
        "average_runtime_per_ticket": batch_runtime / len(tickets),
        "retries": total_retries,
    }

    return pd.DataFrame(results), summary


if __name__ == "__main__":
    from src.data import load_tickets

    tickets = load_tickets()
    predictions, metrics = classify_tickets(tickets)

    predictions.to_csv("outputs/voorspellingen_nano.csv", index=False)

    with open("outputs/run_metrics_nano.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"\nSaved {len(predictions)} predictions")
    print("\nRun statistics:")
    print(f"Model: {metrics['model']}")
    print(f"API calls: {metrics['api_calls']}")
    print(f"Input tokens: {metrics['input_tokens']}")
    print(f"Output tokens: {metrics['output_tokens']}")
    print(f"Total tokens: {metrics['total_tokens']}")
    print(f"API runtime: {metrics['api_runtime_seconds']:.2f}s")
    print(f"Wall-clock runtime: {metrics['wall_clock_runtime_seconds']:.2f}s")
    print(
        f"Average runtime/ticket: {metrics['average_runtime_per_ticket']:.2f}s")
