# Trader Archetypes Plan

## Goal

Implement a very simple, notebook-friendly analysis for **TRUMP coin trader
archetypes**.

The only goals are:

- define 3 trader types that primarily trade TRUMP coin
  - rational arbitrageurs
  - manipulators
  - herd-following retail
- compute Shapley values for all available market features for each trader type
- explain the top 3 most important features for each trader type

This is intentionally a lightweight behavioral analysis on top of the existing
TRUMP processed dataset. It is not a new market simulator and it is not a new
RL environment.

## What Was Implemented

The main implementation lives in
`/Users/matteoperona/Classes/RL/trumpcoin-rl/trader_archetypes.py`.

It includes:

- loading the existing processed TRUMP dataset
- simple rule-based TRUMP trader archetypes
- trader score generation and action labeling
- exact Shapley value computation across the 9 current features
- top-3 feature explanation helpers
- a small action-summary table for notebook inspection
- smoke tests that can be run from Jupyter

## Simplifying Assumptions

- All three archetypes are modeled as **TRUMP-focused decision rules** on the
  existing feature table.
- The analysis explains each trader's **decision score**, not actual learned
  policy gradients or market impact.
- The trader rules are hand-crafted and interpretable, not trained.

This makes the output easy to inspect and defend in a notebook.

## Trader Types

### Rational Arbitrageur

- Modeled as a mean-reversion trader
- More likely to buy after sharp short-term drops and reduce exposure after
  short-term rallies
- Penalized by volatility

### Manipulator

- Modeled as a trader sensitive to volume spikes, sharp short-term moves, and
  timing effects
- Intended to approximate front-running / wash-trade style pressure

### Herd-Following Retail

- Modeled as a momentum chaser
- More likely to buy into recent positive returns and high attention / volume

## Notebook Workflow

The notebook should do the following:

1. Import `trader_archetypes.py`
2. Run `run_smoke_tests()` to validate the pipeline
3. Build the analysis bundle with `notebook_summary_bundle()`
4. Inspect the trader profile and action summary tables
5. Compute and view Shapley summaries
6. Print the top 3 features for each trader type
7. Read the explanation text for those top features

## Key Functions

- `load_feature_dataset(dataset="trump")`
- `get_default_traders()`
- `compute_trader_profiles(feature_frame, traders=None)`
- `summarize_trader_actions(profiles, traders=None)`
- `compute_trader_shapley_summary(feature_frame, traders=None, sample_size=32, random_state=7)`
- `top_features_for_trader(shapley_summary, trader_name, top_n=3)`
- `explain_top_features(shapley_summary, traders=None, top_n=3)`
- `run_smoke_tests(dataset="trump")`
- `notebook_summary_bundle(dataset="trump", shapley_sample_size=32, random_state=7)`

## How To Use From Jupyter

From the repo root:

```python
from trader_archetypes import (
    notebook_summary_bundle,
    run_smoke_tests,
    top_features_for_trader,
)

tests = run_smoke_tests()
display(tests)

bundle = notebook_summary_bundle(dataset="trump", shapley_sample_size=32)

display(bundle["profiles"].head())
display(bundle["action_summary"])

for trader in ["rational_arbitrageur", "manipulator", "herd_retail"]:
    display(top_features_for_trader(bundle["shapley_summary"], trader))
    for line in bundle["explanations"][trader]:
        print("-", line)
```

## Expected Review Points

When reviewing the notebook output, the main questions should be:

- Do the trader rules feel directionally plausible for TRUMP coin?
- Do the Shapley rankings tell a coherent story for each trader type?
- Do the top-3 explanations read naturally and match the rankings?

## Suggested Next Step

If this looks good, the next step would be to improve the trader formulas or the
feature set, not to build a larger simulator.
