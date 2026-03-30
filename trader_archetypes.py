from __future__ import annotations

import math
import os
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "feature_log_return_1h",
    "feature_log_return_4h",
    "feature_log_return_24h",
    "feature_volatility_24h",
    "feature_log_volume",
    "feature_hour_sin",
    "feature_hour_cos",
    "feature_day_sin",
    "feature_day_cos",
]

FEATURE_LABELS = {
    "feature_log_return_1h": "1h Return",
    "feature_log_return_4h": "4h Return",
    "feature_log_return_24h": "24h Return",
    "feature_volatility_24h": "24h Volatility",
    "feature_log_volume": "Log Volume",
    "feature_hour_sin": "Hour Sin",
    "feature_hour_cos": "Hour Cos",
    "feature_day_sin": "Day Sin",
    "feature_day_cos": "Day Cos",
}

FEATURE_EXPLANATIONS = {
    "feature_log_return_1h": "very recent momentum",
    "feature_log_return_4h": "short-horizon trend",
    "feature_log_return_24h": "day-scale direction",
    "feature_volatility_24h": "market stress and risk",
    "feature_log_volume": "market attention and activity",
    "feature_hour_sin": "intraday timing",
    "feature_hour_cos": "intraday timing",
    "feature_day_sin": "day-of-week timing",
    "feature_day_cos": "day-of-week timing",
}

TRADER_ORDER = [
    "rational_arbitrageur",
    "manipulator",
    "herd_retail",
]


@dataclass(frozen=True)
class TraderArchetype:
    name: str
    display_name: str
    weights: Mapping[str, float]
    bias: float
    positive_action_label: str
    negative_action_label: str
    buy_threshold: float = 0.25
    sell_threshold: float = -0.25

    def score_from_mapping(self, feature_values: Mapping[str, float]) -> float:
        linear = self.bias
        for feature, weight in self.weights.items():
            linear += weight * float(feature_values[feature])
        return math.tanh(linear)

    def score_frame(self, feature_frame: pd.DataFrame) -> pd.Series:
        linear = np.full(len(feature_frame), self.bias, dtype=np.float64)
        for feature, weight in self.weights.items():
            linear += weight * feature_frame[feature].to_numpy(dtype=np.float64)
        return pd.Series(np.tanh(linear), index=feature_frame.index, name=f"{self.name}_score")

    def actions_from_scores(self, scores: pd.Series) -> pd.Series:
        actions = np.where(
            scores >= self.buy_threshold,
            self.positive_action_label,
            np.where(scores <= self.sell_threshold, self.negative_action_label, "hold"),
        )
        return pd.Series(actions, index=scores.index, name=f"{self.name}_action")


def get_default_traders() -> Dict[str, TraderArchetype]:
    return {
        "rational_arbitrageur": TraderArchetype(
            name="rational_arbitrageur",
            display_name="Rational Arbitrageur",
            weights={
                "feature_log_return_1h": -1.10,
                "feature_log_return_4h": -0.85,
                "feature_log_return_24h": -0.45,
                "feature_volatility_24h": -0.55,
                "feature_log_volume": 0.20,
                "feature_hour_cos": 0.10,
            },
            bias=0.0,
            positive_action_label="buy_mispricing",
            negative_action_label="sell_rally",
        ),
        "manipulator": TraderArchetype(
            name="manipulator",
            display_name="Manipulator",
            weights={
                "feature_log_return_1h": 0.75,
                "feature_log_return_4h": 0.35,
                "feature_volatility_24h": 0.65,
                "feature_log_volume": 1.10,
                "feature_hour_sin": -0.20,
                "feature_hour_cos": 0.25,
                "feature_day_cos": -0.15,
            },
            bias=0.10,
            positive_action_label="pump_or_front_run",
            negative_action_label="fade_or_dump",
        ),
        "herd_retail": TraderArchetype(
            name="herd_retail",
            display_name="Herd-Following Retail",
            weights={
                "feature_log_return_1h": 0.95,
                "feature_log_return_4h": 0.80,
                "feature_log_return_24h": 0.40,
                "feature_volatility_24h": 0.20,
                "feature_log_volume": 0.70,
                "feature_day_sin": 0.15,
            },
            bias=0.0,
            positive_action_label="chase_uptrend",
            negative_action_label="panic_sell",
        ),
    }


def _repo_data_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "processed")


def _dataset_filename(dataset: str) -> str:
    dataset = dataset.lower()
    return "rl_dataset_btc_hourly.csv" if dataset == "btc" else "rl_dataset_hourly.csv"


def load_feature_dataset(dataset: str = "trump") -> pd.DataFrame:
    source = pd.read_csv(
        os.path.join(_repo_data_dir(), _dataset_filename(dataset)),
        index_col=0,
        parse_dates=True,
    )

    feature_frame = pd.DataFrame(index=source.index)
    for feature in FEATURE_COLUMNS:
        raw = source[feature].astype(float)
        std = raw.std()
        feature_frame[feature] = (raw - raw.mean()) / (std if std > 1e-8 else 1.0)
        feature_frame[f"raw__{feature}"] = raw

    close_col = "BTC-USD_Close" if dataset.lower() == "btc" else "TRUMP35336-USD_Close"
    feature_frame["raw__close"] = source[close_col].astype(float)
    return feature_frame


def summarize_trader_actions(
    profiles: pd.DataFrame,
    traders: Mapping[str, TraderArchetype] | None = None,
) -> pd.DataFrame:
    traders = traders or get_default_traders()
    rows: List[Dict[str, float | str | int]] = []
    for trader_name in TRADER_ORDER:
        trader = traders[trader_name]
        action_counts = profiles[f"{trader_name}_action"].value_counts().to_dict()
        for action, count in action_counts.items():
            rows.append({
                "trader": trader_name,
                "display_name": trader.display_name,
                "action": action,
                "count": int(count),
                "share": float(count / max(len(profiles), 1)),
            })
    return pd.DataFrame(rows).sort_values(["trader", "count"], ascending=[True, False]).reset_index(drop=True)


def compute_trader_profiles(
    feature_frame: pd.DataFrame,
    traders: Mapping[str, TraderArchetype] | None = None,
) -> pd.DataFrame:
    traders = traders or get_default_traders()
    profile = pd.DataFrame(index=feature_frame.index)
    for name in TRADER_ORDER:
        trader = traders[name]
        scores = trader.score_frame(feature_frame[FEATURE_COLUMNS])
        profile[f"{name}_score"] = scores
        profile[f"{name}_action"] = trader.actions_from_scores(scores)
    return profile


def _coalition_weight(num_features: int, coalition_size: int) -> float:
    return (
        math.factorial(coalition_size)
        * math.factorial(num_features - coalition_size - 1)
        / math.factorial(num_features)
    )


def exact_shapley_for_row(
    trader: TraderArchetype,
    row: Mapping[str, float],
    baseline: Mapping[str, float],
    features: Sequence[str] | None = None,
) -> Dict[str, float]:
    features = tuple(features or FEATURE_COLUMNS)
    shapley_values = {feature: 0.0 for feature in features}

    for feature in features:
        other_features = [f for f in features if f != feature]
        for coalition_size in range(len(other_features) + 1):
            for coalition in combinations(other_features, coalition_size):
                without_feature = dict(baseline)
                with_feature = dict(baseline)

                for coalition_feature in coalition:
                    without_feature[coalition_feature] = row[coalition_feature]
                    with_feature[coalition_feature] = row[coalition_feature]

                with_feature[feature] = row[feature]
                marginal = (
                    trader.score_from_mapping(with_feature)
                    - trader.score_from_mapping(without_feature)
                )
                shapley_values[feature] += _coalition_weight(len(features), coalition_size) * marginal

    return shapley_values


def compute_trader_shapley_summary(
    feature_frame: pd.DataFrame,
    traders: Mapping[str, TraderArchetype] | None = None,
    sample_size: int = 32,
    random_state: int = 7,
) -> pd.DataFrame:
    traders = traders or get_default_traders()
    sample_size = min(sample_size, len(feature_frame))

    sampled = feature_frame[FEATURE_COLUMNS].sample(n=sample_size, random_state=random_state)
    baseline = feature_frame[FEATURE_COLUMNS].mean().to_dict()

    rows: List[Dict[str, float | str]] = []
    for trader_name in TRADER_ORDER:
        trader = traders[trader_name]
        per_feature_values = {feature: [] for feature in FEATURE_COLUMNS}

        for _, row in sampled.iterrows():
            shapley = exact_shapley_for_row(trader, row.to_dict(), baseline)
            for feature, value in shapley.items():
                per_feature_values[feature].append(value)

        for feature, values in per_feature_values.items():
            values_arr = np.asarray(values, dtype=np.float64)
            rows.append({
                "trader": trader_name,
                "display_name": trader.display_name,
                "feature": feature,
                "feature_label": FEATURE_LABELS[feature],
                "mean_shapley": float(values_arr.mean()),
                "mean_abs_shapley": float(np.abs(values_arr).mean()),
            })

    summary = pd.DataFrame(rows)
    summary["rank"] = summary.groupby("trader")["mean_abs_shapley"].rank(
        method="dense",
        ascending=False,
    )
    return summary.sort_values(["trader", "rank", "feature"]).reset_index(drop=True)


def top_features_for_trader(
    shapley_summary: pd.DataFrame,
    trader_name: str,
    top_n: int = 3,
) -> pd.DataFrame:
    subset = shapley_summary[shapley_summary["trader"] == trader_name]
    return subset.nsmallest(top_n, "rank").copy()


def explain_top_features(
    shapley_summary: pd.DataFrame,
    traders: Mapping[str, TraderArchetype] | None = None,
    top_n: int = 3,
) -> Dict[str, List[str]]:
    traders = traders or get_default_traders()
    explanations: Dict[str, List[str]] = {}

    for trader_name in TRADER_ORDER:
        trader = traders[trader_name]
        top_df = top_features_for_trader(shapley_summary, trader_name, top_n=top_n)
        lines = []
        for _, row in top_df.iterrows():
            direction = "pushes the trader toward more aggressive action" if row["mean_shapley"] >= 0 else "pushes the trader toward restraint or reversal"
            lines.append(
                f"{row['feature_label']} matters because it captures {FEATURE_EXPLANATIONS[row['feature']]}; "
                f"on average it {direction} for {trader.display_name.lower()}."
            )
        explanations[trader_name] = lines

    return explanations


def run_smoke_tests(dataset: str = "trump") -> pd.DataFrame:
    results = []

    feature_frame = load_feature_dataset(dataset)
    results.append({
        "test": "dataset_loads",
        "passed": len(feature_frame) > 100 and all(col in feature_frame.columns for col in FEATURE_COLUMNS),
        "detail": f"rows={len(feature_frame)}, features={len(FEATURE_COLUMNS)}",
    })

    traders = get_default_traders()
    profiles = compute_trader_profiles(feature_frame.head(32), traders)
    expected_profile_cols = {f"{name}_score" for name in TRADER_ORDER} | {f"{name}_action" for name in TRADER_ORDER}
    results.append({
        "test": "trader_profiles_build",
        "passed": expected_profile_cols.issubset(set(profiles.columns)),
        "detail": f"profile_columns={len(profiles.columns)}",
    })

    shapley_summary = compute_trader_shapley_summary(feature_frame.head(64), traders, sample_size=4, random_state=1)
    results.append({
        "test": "shapley_summary_builds",
        "passed": len(shapley_summary) == len(FEATURE_COLUMNS) * len(TRADER_ORDER),
        "detail": f"rows={len(shapley_summary)}",
    })

    action_summary = summarize_trader_actions(profiles, traders)
    results.append({
        "test": "action_summary_builds",
        "passed": len(action_summary) > 0 and set(["trader", "action", "count", "share"]).issubset(set(action_summary.columns)),
        "detail": f"rows={len(action_summary)}",
    })

    return pd.DataFrame(results)


def notebook_summary_bundle(
    dataset: str = "trump",
    shapley_sample_size: int = 32,
    random_state: int = 7,
) -> Dict[str, object]:
    feature_frame = load_feature_dataset(dataset)
    traders = get_default_traders()
    profiles = compute_trader_profiles(feature_frame, traders)
    shapley_summary = compute_trader_shapley_summary(
        feature_frame,
        traders=traders,
        sample_size=shapley_sample_size,
        random_state=random_state,
    )
    explanations = explain_top_features(shapley_summary, traders=traders, top_n=3)
    action_summary = summarize_trader_actions(profiles, traders=traders)
    return {
        "feature_frame": feature_frame,
        "profiles": profiles,
        "action_summary": action_summary,
        "shapley_summary": shapley_summary,
        "explanations": explanations,
    }


if __name__ == "__main__":
    bundle = notebook_summary_bundle()
    print(run_smoke_tests().to_string(index=False))
    print("\nTop feature ranks:")
    for trader_name in TRADER_ORDER:
        print(f"\n{trader_name}")
        print(top_features_for_trader(bundle["shapley_summary"], trader_name).to_string(index=False))
