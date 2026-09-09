from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/tickets.csv")

REQUIRED_COLUMNS = {
    "ticket_id",
    "onderwerp",
    "omschrijving",
    "systeemmelding",
    "interne_notitie",
}


def load_tickets(path=DATA_PATH):
    """Load, clean and validate the ticket dataset"""
    df = pd.read_csv(path)

    validate_columns(df)

    # Remove completely empty rows and exact duplicates
    df = df.dropna(how="all")
    df = df.drop_duplicates()

    validate_tickets(df)

    return df


def validate_columns(df):
    """Check whether the expected columns are present"""
    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def validate_tickets(df):
    """Validate the cleaned ticket dataset"""

    if df["ticket_id"].isna().any():
        raise ValueError("Missing ticket IDs found")

    if not df["ticket_id"].is_unique:
        raise ValueError("Duplicate ticket IDs found")

    # Assignment specifically mentions 100 unique tickets
    if len(df) != 100:
        raise ValueError(f"Expected 100 tickets, found {len(df)}")


if __name__ == "__main__":
    tickets = load_tickets()

    print(f"Loaded {len(tickets)} valid tickets.")
    print(tickets.head())
