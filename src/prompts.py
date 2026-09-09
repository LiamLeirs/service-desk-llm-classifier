import pandas as pd


def format_ticket(ticket: pd.Series) -> str:
    """Format the relevant ticket fields for the LLM."""

    fields = {
        "Onderwerp": ticket["onderwerp"],
        "Omschrijving": ticket["omschrijving"],
        "Systeemmelding": ticket["systeemmelding"],
        "Interne notitie": ticket["interne_notitie"],
    }

    lines = []

    for name, value in fields.items():
        if pd.notna(value):
            lines.append(f"{name}: {value}")

    return "\n".join(lines)


if __name__ == "__main__":
    from src.data import load_tickets

    tickets = load_tickets()

    print(format_ticket(tickets.iloc[0]))
