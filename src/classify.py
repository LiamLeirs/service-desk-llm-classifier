import os
from dotenv import load_dotenv
from openai import OpenAI
from enum import Enum
from pydantic import BaseModel
from src.prompts import SYSTEM_PROMPT, format_ticket
import pandas as pd


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


def classify_ticket(ticket) -> ClassificationResult:
    response = client.responses.parse(
        model=MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_ticket(ticket)},
        ],
        text_format=ClassificationResult,
    )

    return response.output_parsed


def classify_tickets(tickets: pd.DataFrame) -> pd.DataFrame:
    results = []

    for _, ticket in tickets.iterrows():
        result = classify_ticket(ticket)

        results.append({
            "ticket_id": ticket["ticket_id"],
            "categorie": result.categorie.value,
            "prioriteit": ticket["prioriteit"],
            "te_valideren": int(result.te_valideren),
            "security_flag": int(result.security_flag),
            "reden": result.reden,
        })

        print(f"{ticket['ticket_id']}: {result.categorie.value}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    from src.data import load_tickets

    tickets = load_tickets()
    predictions = classify_tickets(tickets)

    predictions.to_csv(
        "outputs/voorspellingen.csv",
        index=False,
    )

    print(f"\nSaved {len(predictions)} predictions")
