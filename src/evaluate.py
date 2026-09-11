import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.models import Category


GOLD_PATH = Path("data/gold_labels.csv")
CATEGORIES = [category.value for category in Category]

MODEL_PRICING = {
    "gpt-5-nano": {
        "input": 0.05,
        "output": 0.35,
    },
    "gpt-5.4": {
        "input": 2.15,
        "output": 12.88,
    },
}


def load_data(predictions_path: Path) -> pd.DataFrame:
    """Load and combine gold labels with model predictions."""
    gold = pd.read_csv(GOLD_PATH)
    predictions = pd.read_csv(predictions_path)

    if not gold["ticket_id"].is_unique:
        raise ValueError("Duplicate ticket IDs found in gold labels")

    if not predictions["ticket_id"].is_unique:
        raise ValueError("Duplicate ticket IDs found in predictions")

    if len(gold) != len(predictions):
        raise ValueError(
            f"Different number of tickets: "
            f"{len(gold)} gold labels vs {len(predictions)} predictions"
        )

    df = gold.merge(
        predictions,
        on="ticket_id",
        how="inner",
        suffixes=("_gold", "_pred"),
    )

    if len(df) != len(gold):
        raise ValueError("Some ticket IDs are missing after merging")

    return df


def load_metrics(metrics_path: Path) -> dict:
    """Load token usage and runtime metrics."""
    with open(metrics_path, encoding="utf-8") as file:
        return json.load(file)


def evaluate_validation_flag(df: pd.DataFrame) -> dict:
    """Evaluate whether te_valideren catches classification errors."""
    incorrect = df["categorie_gold"] != df["categorie_pred"]
    flagged = df["te_valideren"] == 1

    errors = int(incorrect.sum())
    errors_flagged = int((incorrect & flagged).sum())

    return {
        "flagged": int(flagged.sum()),
        "errors": errors,
        "errors_flagged": errors_flagged,
        "error_detection_rate": errors_flagged / errors if errors else 0.0,
    }


def evaluate_cost(metrics: dict) -> dict:
    """Calculate estimated API cost from token usage."""
    model = metrics["model"]

    if model not in MODEL_PRICING:
        raise ValueError(f"No pricing configured for model: {model}")

    pricing = MODEL_PRICING[model]
    input_cost = metrics["input_tokens"] / 1_000_000 * pricing["input"]
    output_cost = metrics["output_tokens"] / 1_000_000 * pricing["output"]
    total_cost = input_cost + output_cost

    return {
        "currency": "EUR",
        "input_price_per_million": pricing["input"],
        "output_price_per_million": pricing["output"],
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
        "cost_per_ticket": total_cost / metrics["api_calls"],
    }


def save_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, output_path: Path) -> None:
    """Create and save the confusion matrix."""
    matrix = confusion_matrix(y_true, y_pred, labels=CATEGORIES)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CATEGORIES,
    )

    fig, ax = plt.subplots(figsize=(11, 9))
    display.plot(ax=ax, xticks_rotation=45, values_format="d")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate(df: pd.DataFrame, metrics: dict, evaluation_path: Path, errors_path: Path, confusion_matrix_path: Path) -> dict:
    """Evaluate model predictions and save the results."""
    y_true = df["categorie_gold"]
    y_pred = df["categorie_pred"]

    accuracy = accuracy_score(y_true, y_pred)
    correct = int((y_true == y_pred).sum())
    errors = df[y_true != y_pred].copy()

    report = classification_report(
        y_true,
        y_pred,
        labels=CATEGORIES,
        zero_division=0,
        output_dict=True,
    )

    errors.to_csv(errors_path, index=False)

    evaluation = {
        "model": metrics["model"],
        "accuracy": accuracy,
        "correct": correct,
        "total": len(df),
        "misclassified": len(errors),
        "classification_report": report,
        "validation": evaluate_validation_flag(df),
        "run_statistics": {
            "api_calls": metrics["api_calls"],
            "input_tokens": metrics["input_tokens"],
            "output_tokens": metrics["output_tokens"],
            "reasoning_tokens": metrics["reasoning_tokens"],
            "visible_output_tokens": metrics["visible_output_tokens"],
            "total_tokens": metrics["total_tokens"],
            "wall_clock_runtime_seconds": metrics["wall_clock_runtime_seconds"],
            "average_runtime_per_ticket": metrics["average_runtime_per_ticket"],
            "retries": metrics["retries"],
        },
        "cost": evaluate_cost(metrics),
    }

    with open(evaluation_path, "w", encoding="utf-8") as file:
        json.dump(evaluation, file, indent=2)

    save_confusion_matrix(y_true, y_pred, confusion_matrix_path)

    return evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["nano", "gpt54"], default="nano")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    predictions_path = Path(f"outputs/voorspellingen_{args.model}.csv")
    metrics_path = Path(f"outputs/run_metrics_{args.model}.json")
    evaluation_path = Path(f"outputs/evaluation_{args.model}.json")
    errors_path = Path(f"outputs/errors_{args.model}.csv")
    confusion_matrix_path = Path(f"outputs/confusion_matrix_{args.model}.png")

    data = load_data(predictions_path)
    metrics = load_metrics(metrics_path)

    evaluation = evaluate(
        data,
        metrics,
        evaluation_path,
        errors_path,
        confusion_matrix_path,
    )

    print(
        f"{metrics['model']}: "
        f"{evaluation['accuracy']:.2%} accuracy, "
        f"{evaluation['misclassified']} errors"
    )
