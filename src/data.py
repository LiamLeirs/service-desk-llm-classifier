from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/tickets.csv")


def load_tickets(path=DATA_PATH):
    return pd.read_csv(path)


def inspect_tickets(df):
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Unique ticket IDs: {df['ticket_id'].nunique()}")

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nRows without ticket ID:")
    print(df[df["ticket_id"].isna()].to_string())

    print("\nDuplicate ticket IDs:")
    duplicates = df[df["ticket_id"].duplicated(keep=False)]
    print(duplicates.to_string())


if __name__ == "__main__":
    tickets = load_tickets()
    inspect_tickets(tickets)
