# Service Desk Ticket Classifier

LLM-based solution for automatically classifying service desk tickets into predefined categories.

Besides the required category, the classifier preserves the original priority and adds flags for tickets that may require manual validation or contain a potential security incident.

## Setup

Create and activate the Conda environment:

```bash
conda create -n ticket-classifier python=3.11
conda activate ticket-classifier
pip install -r requirements.txt
```

Create a `.env` file based on `.env.example`:

```env
LLM_API_KEY=
AZURE_OPENAI_BASE_URL=
GPT_54_DEPLOYMENT=
GPT_NANO_DEPLOYMENT=
```

## Usage

Run the classifier with GPT-5.4:

```bash
python -m src.classify --model gpt54
```

Or with GPT-5-nano:

```bash
python -m src.classify --model nano
```

Evaluate a run using:

```bash
python -m src.evaluate --model gpt54
python -m src.evaluate --model nano
```

The final predictions are stored in:

```text
outputs/voorspellingen.csv
```

## Approach and design choices

I approached the problem as a zero-shot classification task using an LLM.

Before classification, the supplied dataset is cleaned and validated. Completely empty rows and exact duplicates are removed, and the presence and uniqueness of ticket IDs are checked.

### Input selection

Only fields that contain information relevant to the classification are sent to the model:

- `onderwerp`
- `omschrijving`
- `systeemmelding`
- `interne_notitie`

Empty fields are omitted from the formatted ticket.

`prioriteit` is deliberately not sent to the LLM. It is already present in the source data and is copied directly to the output. Asking the model to reproduce it would introduce unnecessary input and the possibility of changing information that is already known.

### Classification

The category definitions and classification rules are included in the system prompt.

I chose prompt-based classification instead of fine-tuning. The dataset contains only 100 unique tickets and no separate labelled training dataset was supplied. Fine-tuning would therefore require creating and maintaining an additional representative training set, while the classification rules can be expressed directly in the prompt.

The LLM response is parsed into a Pydantic model instead of processing free-form output. The category is represented by an enum, ensuring that the model can only return one of the predefined categories.

### Additional output

Besides the required `ticket_id`, `categorie` and `prioriteit`, I added:

- `te_valideren`: indicates that manual validation may be useful;
- `security_flag`: indicates a suspected security incident;
- `reden`: short explanation of why the category was selected.

I kept the security flag separate because a possible security incident deserves additional attention independently of the normal ticket classification.

The explanation was retained because it makes predictions easier to inspect and was useful when analysing misclassifications.

## Evaluation

Because no gold labels were supplied, I manually assigned a reference category to each of the 100 unique tickets. These gold labels are stored separately from the model predictions.

Accuracy is used as the primary metric because each ticket receives exactly one category. I additionally calculated precision, recall and F1-score per category and generated confusion matrices to inspect which categories are confused with each other.

All model configurations were evaluated against the same gold labels.

| Model                          | Accuracy |  Macro F1 | Incorrect |
| ------------------------------ | -------: | --------: | --------: |
| GPT-5.4                        |  **96%** | **0.962** |     **4** |
| GPT-5-nano                     |      92% |     0.919 |         8 |
| GPT-5-nano (minimal reasoning) |      88% |     0.879 |        12 |

GPT-5.4 achieved the best result and was therefore selected for the final predictions.

### GPT-5.4 error analysis

GPT-5.4 classified 96 of the 100 tickets according to the manually created gold labels. The four remaining cases were inspected manually.

| Ticket   | Gold                    | Prediction         | Assessment |
| -------- | ----------------------- | ------------------ | ---------- |
| INC-4127 | Netwerk                 | Overig             | Ambiguous  |
| INC-4153 | Toegang & Accounts      | Printer            | Ambiguous  |
| INC-4232 | Applicatieondersteuning | Software           | Ambiguous  |
| INC-4284 | Applicatieondersteuning | Toegang & Accounts | Ambiguous  |

These errors illustrate that some category boundaries are subjective rather than representing completely unrelated predictions.

**INC-4127** concerns an outage at the user's private internet provider. `Netwerk` is the gold label because the underlying problem is network connectivity, while `Overig` is also defensible because the outage is outside the company's IT environment.

**INC-4153** concerns pull-print authentication failing because the user is no longer linked to a badge ID. `Toegang & Accounts` was selected as the gold label because authentication is the underlying issue, while the model selected `Printer` because the visible problem occurs entirely in the printing workflow.

**INC-4232** concerns a legitimate calculation tool being quarantined by antivirus software. `Applicatieondersteuning` was selected as the gold label because the business application can no longer be used. `Software` is nevertheless plausible because the immediate cause is endpoint/antivirus software blocking the executable.

**INC-4284** concerns webshop orders no longer reaching the ERP because an API key for the integration expired. `Applicatieondersteuning` was selected because the failure occurs in an application integration. The model selected `Toegang & Accounts`, which is understandable because the direct technical cause is an expired authentication credential.

The 96% accuracy should therefore be interpreted together with the confusion matrix and error analysis rather than as the only measure of quality.

### Validation flag

I also evaluated whether `te_valideren` could be used as an uncertainty mechanism.

The flag did not reliably identify the actual classification errors. For GPT-5.4, eight tickets were flagged for validation, but none of the four misclassified tickets were among them.

I would therefore not use `te_valideren` as an automatic confidence measure in production in its current form. A possible improvement would be a separate validation step or additional confidence signals rather than relying on the same classification call to estimate its own uncertainty.

## Cost and performance

Token usage, API calls and runtime were measured during classification.

| Metric                 |   GPT-5.4 | GPT-5-nano | Nano (minimal reasoning) |
| ---------------------- | --------: | ---------: | -----------------------: |
| API calls              |       100 |        100 |                      100 |
| Input tokens           |   115,626 |    115,626 |                  115,626 |
| Output tokens          |     6,056 |    105,847 |                    8,051 |
| Reasoning tokens       |         0 |     96,448 |                        0 |
| Total tokens           |   121,682 |    221,473 |                  123,677 |
| Runtime                |  276.96 s |   635.30 s |                 272.55 s |
| Average runtime/ticket |    2.77 s |     6.35 s |                   2.73 s |
| Estimated cost         |   €0.3266 |    €0.0428 |                  €0.0086 |
| Estimated cost/ticket  | €0.003266 |  €0.000428 |                €0.000086 |

Costs are estimates based on the applicable Azure OpenAI token prices used during evaluation.

### Cost considerations

Several choices were made to control API usage and cost.

Only relevant ticket fields are included in the prompt and empty values are omitted. All required model-generated outputs are returned in a single structured API call per ticket.

I also compared GPT-5.4 with GPT-5-nano to determine whether a cheaper model could provide sufficient classification quality.

The default GPT-5-nano run produced 96,448 reasoning tokens. Despite being a cheaper model, this resulted in 221,473 total tokens and an average runtime of 6.35 seconds per ticket.

I therefore tested GPT-5-nano with minimal reasoning effort. In this run no reasoning tokens were reported, reducing total usage to 123,677 tokens and the estimated cost for all 100 tickets from approximately €0.0428 to €0.0086. Runtime also decreased from approximately 635 seconds to 273 seconds.

This optimization introduced a quality trade-off: accuracy decreased from 92% to 88%.

GPT-5.4 was considerably more expensive at approximately €0.3266 for the complete dataset, but achieved the highest accuracy at 96%. Since the absolute cost for processing 100 tickets remained low, I selected GPT-5.4 for the final predictions rather than optimizing purely for the lowest API cost.

## Output

The final output contains:

| Column          | Description                                       |
| --------------- | ------------------------------------------------- |
| `ticket_id`     | Unique ticket identifier                          |
| `categorie`     | Predicted category                                |
| `prioriteit`    | Original priority from the source data            |
| `te_valideren`  | Indicates whether manual validation may be useful |
| `security_flag` | Indicates a suspected security incident           |
| `reden`         | Short explanation for the classification          |

## Limitations and possible improvements

The evaluation set is small and the gold labels were created manually, so some category boundaries remain subjective.

The results also show that reducing reasoning effort can lower cost significantly but may reduce classification accuracy.

The `te_valideren` mechanism currently does not reliably identify incorrect classifications and would require further development before production use.

With more labelled historical tickets, future work could include evaluating the solution on a separate held-out test set, improving uncertainty estimation, or comparing the prompt-based approach with a fine-tuned classifier.

## Conclusion

GPT-5.4 achieved the highest classification accuracy at 96%, compared with 92% for GPT-5-nano and 88% for GPT-5-nano with minimal reasoning.

GPT-5-nano demonstrated that the workload can be processed at substantially lower API cost, while the minimal-reasoning experiment showed an additional cost reduction at the expense of accuracy.

For the final predictions, GPT-5.4 was selected because it provided the best classification quality while the absolute processing cost for the complete dataset remained low.
