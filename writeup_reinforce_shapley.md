# Extending The TRUMP Coin RL Study With REINFORCE And Shapley-Based Trader Archetypes

**Matteo Perona**

---

## 1. Introduction

This report extends the earlier TRUMP coin reinforcement learning writeup in
two directions.

First, it adds a policy-gradient agent based on **REINFORCE with a learned
value baseline**. This gives a third RL approach to compare with the existing
Tabular Q-Learning and DDQN agents. The main question is whether an on-policy
policy-gradient method can avoid the failure modes that appeared in the
value-based agents.

Second, it adds a lightweight interpretability layer through a
**trader-archetypes notebook**. Instead of training a new market simulator, the
notebook defines three hand-crafted TRUMP-focused trader types and explains
their decisions with exact Shapley values over the existing feature set.

Together, these additions broaden the project in two useful ways: REINFORCE
tests whether a different RL family behaves better, and the Shapley notebook
tests whether the existing feature table supports clear, interpretable
behavioral stories.

---

## 2. REINFORCE Implementation And Results

### 2.1 Implementation Summary

The REINFORCE implementation lives primarily in `reinforce_agent.py`. It uses a
standard policy-gradient setup, but with a few practical modifications to make
training less unstable in this environment.

**Core design**

| Component | Implementation |
| --- | --- |
| Policy model | Two-hidden-layer MLP mapping the 10-dim observation to logits over 5 discrete actions |
| Baseline model | Separate value network with the same hidden structure |
| Policy update | REINFORCE objective using discounted returns minus the value baseline |
| Variance control | Advantage normalization and gradient clipping |
| Optimizers | Separate Adam optimizers for policy and value networks |
| Data split | 70% train / 30% test, matching the earlier experiments |
| Evaluation policy | Deterministic argmax over policy logits |

The observation space is the same one used by the earlier agents: 9 normalized
market features plus the current portfolio position. The action space is also
unchanged:

1. Heavy Sell
2. Light Sell
3. Hold
4. Light Buy
5. Heavy Buy

The final figure run,
`reinforce_trump_2026-03-26_151936`, used the following hyperparameters:

| Hyperparameter | Value |
| --- | --- |
| Policy learning rate | `1e-4` |
| Value learning rate | `1e-3` |
| Discount factor | `0.99` |
| Hidden dimension | `64` |
| Episodes | `200` |
| Episode window | `24` hours |
| Risk aversion | `0.5` |
| Transaction cost | `0.001` |
| Entropy coefficient | `0.0` |

### 2.2 What Had To Be Fixed

REINFORCE was not just a matter of swapping in a new agent class. The saved run
history shows several concrete failure modes.

**Table 1: REINFORCE implementation problems, fixes, and status**

| Problem | What was changed | Outcome | Status |
| --- | --- | --- | --- |
| High-variance policy-gradient updates | Added a learned value baseline, normalized advantages, and gradient clipping | Training became numerically stable enough to finish consistently | Partly solved |
| Agent initially sees only zero-position states | Randomized the initial training position before each episode | The policy is exposed to buy- and sell-side transitions during training | Solved |
| Full-history training is too non-stationary for on-policy updates | Sampled random contiguous windows from the train split with `sample_train_window()` | Training became more localized and tunable through `episode_length` | Partly solved |
| Policy collapse to extreme actions | Tested different episode windows and episode counts | Collapse was reduced in some runs, but not eliminated | Unsolved |
| Poor generalization to the bearish held-out test period | Compared multiple saved runs on the same test split | Every active REINFORCE policy still lost money out of sample | Unsolved |

Two implementation choices were especially important.

First, the code explicitly breaks the **zero-position symmetry** at reset by
assigning a random starting position during training. Without that, a policy can
easily discover that sell actions from an all-cash state are "safe," never
experience the consequences of holding TRUMP, and never learn meaningful
buy-side behavior.

Second, the code trains on **sampled contiguous windows** rather than always
walking the full training period. That is a better fit for REINFORCE because
the update is on-policy and depends directly on the return collected under the
current policy. In a long, non-stationary crypto series, shorter windows reduce
the chance that one episode mixes together many incompatible market regimes.

### 2.3 Run Sweep And Failure Modes

The saved REINFORCE runs from March 25-26, 2026 show that the agent did not
settle into one stable behavior. Instead, it oscillated between different forms
of collapse depending on the training window and randomness.

**Table 2: Representative REINFORCE runs**

| Run ID | Episodes | Window | Final Portfolio | Reward | Dominant behavior |
| --- | --- | --- | --- | --- | --- |
| `reinforce_trump_2026-03-25_125119` | 50 | earlier config | \$3,590.90 | -1.024 | Heavy-buy collapse |
| `reinforce_trump_2026-03-26_150533` | 100 | 168h | \$10,000.00 | 0.000 | Sell-to-cash collapse |
| `reinforce_trump_2026-03-26_151312` | 200 | 25h | \$9,610.29 | -0.040 | Mostly sell-side, very low exposure |
| `reinforce_trump_2026-03-26_151536` | 200 | 24h | \$8,974.70 | -0.108 | Light-sell dominated |
| `reinforce_trump_2026-03-26_151936` | 200 | 24h | \$5,251.24 | -0.644 | Mostly hold with some light buying |

This sweep is important because it shows that REINFORCE's remaining problem is
not just low performance. It is also **instability across runs**.

Longer windows such as 168 hours often pushed the policy toward a nearly pure
cash-preservation strategy, similar to Q-Learning. The earliest saved run, by
contrast, was heavily buy-biased and finished with only \$3,590.90. Shorter
24-25 hour windows reduced some of this collapse and occasionally produced
moderate, mixed policies, but even then the test-set outcomes remained highly
sensitive.

In other words, the implementation fixes made REINFORCE trainable, but they did
not make it reliable.

### 2.4 Training Behavior

The final figure run is much flatter than DDQN and much less catastrophic at the
start than Q-Learning.

![REINFORCE Training Curve](images/reinforce-training.png)

For `reinforce_trump_2026-03-26_151936`, the 200-episode training reward series
had:

- mean reward `-0.0104`
- minimum reward `-0.3041`
- maximum reward `0.2533`
- first-10-episode mean `-0.0082`
- last-10-episode mean `-0.0246`

That is a useful contrast with the earlier agents. Q-Learning moved from a very
negative early regime toward zero reward, while DDQN pushed upward into strongly
positive training reward. REINFORCE mostly oscillated near zero. This suggests
that the baseline and shorter windows reduced variance, but they did not reveal
a strong exploitable signal in the training data.

The training comparison figure makes that difference clear.

![Training Comparison](images/training-comparison.png)

Using the runs shown in the figure:

| Agent | Run ID | First-10 Mean Reward | Last-10 Mean Reward | Interpretation |
| --- | --- | --- | --- | --- |
| Q-Learning | `q_learning_trump_2026-03-05_135405` | -2.006 | -0.035 | Learns to avoid exposure |
| DDQN | `dqn_trump_2026-03-04_113142` | -2.071 | 3.333 | Strong fit to training set |
| REINFORCE | `reinforce_trump_2026-03-26_151936` | -0.008 | -0.025 | Stable but flat |

### 2.5 Test-Set Performance Compared With Q-Learning And DDQN

The evaluation figure for the selected REINFORCE run is shown below.

![REINFORCE Evaluation](images/reinforce-results.png)

On the test set, the selected REINFORCE run finished with a portfolio value of
**\$5,251.24** and a total reward of **-0.6441**. Its action distribution was:

- Heavy Sell: 56
- Light Sell: 4
- Hold: 2,359
- Light Buy: 442
- Heavy Buy: 33

Out of 2,894 evaluation steps, that means the policy chose:

- `81.5%` Hold
- `15.3%` Light Buy
- `1.9%` Heavy Sell
- `1.1%` Heavy Buy
- `0.1%` Light Sell

So the final REINFORCE behavior is not a one-action collapse, but it is still
highly conservative. It mostly waits, occasionally adds modest exposure, and
almost never takes large reallocations.

**Table 3: Test-set comparison on TRUMP**

| Agent | Run ID | Final Portfolio | Reward | Behavioral summary |
| --- | --- | --- | --- | --- |
| Q-Learning | `q_learning_trump_2026-03-05_135405` | **\$10,000.00** | **0.0000** | Cash-preservation policy; no buy actions |
| DDQN (regularized figure run) | `dqn_trump_2026-03-04_113142` | \$4,148.14 | -0.8800 | Collapsed to heavy buying |
| DDQN (best active run from prior report) | `dqn_no_reg_trump_2026-03-03_184004` | **\$5,650.39** | **-0.5710** | Most diverse active trading policy |
| REINFORCE | `reinforce_trump_2026-03-26_151936` | \$5,251.24 | -0.6441 | Mostly hold with small buy bias |

This comparison puts REINFORCE in a middle position.

It clearly improves on the **regularized DDQN figure run**, which overfit
training and collapsed to a 77% Heavy Buy test policy. REINFORCE's more
conservative hold/light-buy mix preserved more capital and achieved a less
negative reward.

However, REINFORCE still does **not** beat **Q-Learning**, whose coarse
discretization once again discovered that avoiding the market entirely is the
safest response to a falling test period. REINFORCE also falls slightly behind
the earlier **unregularized DDQN** run, which remained the best active trading
agent in the previous report.

### 2.6 What REINFORCE Added To The Project

The main contribution of the REINFORCE implementation is not that it solved the
TRUMP coin trading problem. It did not. Its value is that it clarifies which
failure modes belong to the earlier agents and which ones are more fundamental.

**Solved or improved**

- The project now includes a true policy-gradient baseline instead of only
  value-based methods.
- The implementation is stable enough to run repeated experiments and save
  comparable artifacts.
- Randomized starting positions and sampled episode windows reduced the most
  trivial exploration problems.
- The final figure run learned a more moderate policy than the heavy-buy DDQN
  collapse.

**Still unsolved**

- REINFORCE remains highly sensitive to episode length and random training
  windows.
- The training curve stays near zero instead of showing clear improvement.
- Test-set generalization is still poor in the held-out downtrend.
- The best REINFORCE runs often preserve capital mainly by reducing exposure,
  not by learning a truly profitable timing strategy.

The deeper lesson is similar to the earlier one from DDQN: changing the RL
algorithm changes the **style of failure**, but it does not remove the
underlying difficulty of learning robust trading behavior from a short,
non-stationary, low-signal crypto time series.

---

## 3. Trader Archetypes And Shapley Value Tests

### 3.1 Goal Of The Notebook

The second extension to the project is the notebook
`output/jupyter-notebook/trader-archetypes-analysis.ipynb`, supported by
`trader_archetypes.py` and summarized in `TRADER_ARCHETYPES_README.md`.

This part of the project is intentionally lightweight. It is **not** a new RL
agent and **not** a new market simulator. Instead, it asks a narrower question:

> If we define a few interpretable TRUMP coin trader archetypes directly on the
> existing feature table, which features matter most to each type?

The notebook defines three rule-based traders:

| Trader | Intuition |
| --- | --- |
| Rational Arbitrageur | Mean-reversion trader that buys dips, sells rallies, and dislikes volatility |
| Manipulator | Trader driven by volume spikes, short-term moves, and timing effects |
| Herd-Following Retail | Momentum chaser that reacts to recent returns and market attention |

Each trader is modeled as a weighted linear score passed through `tanh`, then
converted into an action label with buy/sell thresholds. The notebook explains
those scores with **exact Shapley values** over the 9 current TRUMP market
features.

### 3.2 How The Shapley Pipeline Works

The implementation in `trader_archetypes.py` proceeds in four stages.

1. Load the processed TRUMP feature dataset and z-score each feature.
2. Score each row for each trader archetype and convert the score to an action.
3. Compute exact per-row Shapley values using the mean feature vector as the
   baseline.
4. Average those values across a sample of rows to build a notebook-friendly
   summary table.

Because there are 9 features, the exact Shapley routine checks all coalitions of
the other 8 features for each feature under analysis. That means the notebook is
doing an exact attribution calculation for the hand-crafted score function,
rather than a heuristic feature-importance estimate.

The notebook uses `shapley_sample_size=32` for its main summary. That keeps the
computation practical while still producing stable-looking rankings.

### 3.3 All Shapley-Related Tests

The project's Shapley tests are currently implemented as smoke tests through
`run_smoke_tests(dataset="trump")`. All of them passed in the notebook.

**Table 4: Notebook smoke tests**

| Test | Result | What it checks |
| --- | --- | --- |
| `dataset_loads` | Pass (`rows=9650, features=9`) | The processed TRUMP dataset loads and contains all 9 feature columns |
| `trader_profiles_build` | Pass (`profile_columns=6`) | The three trader scores and three trader action columns are created correctly |
| `shapley_summary_builds` | Pass (`rows=27`) | The exact Shapley pipeline runs and returns `3 traders x 9 features` |
| `action_summary_builds` | Pass (`rows=9`) | The notebook action-summary table is produced correctly |

These are important tests, but they should be described honestly: they are
**pipeline validation tests**, not formal proofs of Shapley correctness. They
verify that the dataset loads, the archetype rules execute, and the exact
Shapley routine returns the expected shaped output without errors. That is
appropriate for a notebook-oriented analysis layer.

One encouraging detail is that several unused features receive exactly zero
mean-absolute Shapley value for the relevant archetypes. That is the pattern we
would expect if the attribution code is respecting the hand-written trader
formulas.

### 3.4 Notebook Output: Action Summaries

The first notebook output is a simple action summary over the full TRUMP feature
table.

**Table 5: Action summary by trader archetype**

| Trader | Most common actions | Interpretation |
| --- | --- | --- |
| Rational Arbitrageur | `buy_mispricing` 43.9%, `sell_rally` 35.4%, `hold` 20.7% | Frequently reacts to short-run reversals |
| Manipulator | `fade_or_dump` 47.2%, `pump_or_front_run` 42.9%, `hold` 10.0% | Most aggressive and least likely to hold |
| Herd-Following Retail | `panic_sell` 49.2%, `chase_uptrend` 36.9%, `hold` 13.9% | Switches quickly between chasing and capitulating |

These action shares are directionally plausible for TRUMP coin.

The rational arbitrageur is the most balanced of the three. The manipulator is
the most active and spends the least time holding. The herd-retail trader has a
strong boom-bust pattern, alternating between chasing and panic selling more
than it stays neutral.

### 3.5 Shapley Rankings

The main result of the notebook is the ranking of features by **mean absolute
Shapley value**.

**Table 6: Top 3 features per trader**

| Trader | Feature | Mean Shapley | Mean Absolute Shapley | Rank |
| --- | --- | --- | --- | --- |
| Rational Arbitrageur | 1h Return | 0.0005 | 0.4374 | 1 |
| Rational Arbitrageur | 4h Return | -0.0270 | 0.2930 | 2 |
| Rational Arbitrageur | 24h Volatility | 0.0355 | 0.1732 | 3 |
| Manipulator | Log Volume | -0.0895 | 0.7067 | 1 |
| Manipulator | 1h Return | -0.0229 | 0.2810 | 2 |
| Manipulator | 24h Volatility | -0.0521 | 0.1805 | 3 |
| Herd-Following Retail | Log Volume | -0.0431 | 0.4889 | 1 |
| Herd-Following Retail | 1h Return | -0.0152 | 0.3758 | 2 |
| Herd-Following Retail | 4h Return | 0.0214 | 0.2783 | 3 |

Several useful patterns appear immediately.

**Rational Arbitrageur.**
The two strongest signals are the 1-hour and 4-hour returns, followed by
24-hour volatility. This is exactly what we would expect from a mean-reversion
archetype: very recent price movement matters most, and market stress still
matters because it changes how aggressively a rational trader should fade a move.

**Manipulator.**
Log volume is by far the dominant feature. That is a coherent result for a
trader archetype meant to react to attention spikes, crowded flow, or
front-running opportunities. The next two features, 1-hour return and
24-hour volatility, reinforce the same story: manipulators care about short-run
movement and conditions where price can be pushed around more easily.

**Herd-Following Retail.**
The most important features are log volume, 1-hour return, and 4-hour return.
That is also directionally plausible. A herd trader reacts to what is visible
right now: attention, recent price movement, and short-horizon trend. The fact
that 4-hour return ranks above 24-hour return supports the interpretation that
retail-style behavior is driven more by recent excitement than by long-horizon
fundamentals.

### 3.6 What The Notebook Adds

This notebook adds something the RL experiments do not provide on their own:
**interpretable feature stories**.

The RL agents can tell us whether a policy trained on historical data performs
well out of sample, but they are not very transparent. The trader-archetype
notebook works in the opposite direction. It starts from transparent behavioral
rules and asks whether the project's feature set can recover intuitive,
coherent importance rankings.

That check is useful for the broader project because it suggests the 9-feature
table is at least expressive enough to distinguish:

- short-horizon mean reversion
- attention- and volume-driven manipulation pressure
- momentum-chasing retail behavior

### 3.7 Limitations

The notebook is intentionally simple, so its limitations should be stated
clearly.

- The archetypes are **hand-crafted**, not learned from data.
- The Shapley analysis explains a trader's **score function**, not real market
  impact or equilibrium behavior.
- The test suite is a set of smoke tests, not a full mathematical validation
  suite for attribution correctness.
- The sign of `mean_shapley` can be sensitive to the sampled rows and the mean
  baseline, so the strongest conclusions should come from
  `mean_abs_shapley` ranking rather than sign alone.

Those limitations do not undermine the notebook's purpose. They simply define
it correctly: this is an interpretable behavioral analysis layer built on top of
the existing TRUMP feature dataset.

---

## 4. Conclusion

These two additions deepen the project in complementary ways.

The REINFORCE implementation broadens the reinforcement learning comparison from
value-based methods alone to include an on-policy policy-gradient baseline. It
successfully introduced a stable new training pipeline with a value baseline,
advantage normalization, randomized starting positions, and sampled training
windows. But the final result is still sobering: REINFORCE did not learn a
robust profitable policy on TRUMP coin. It mostly learned a cautious,
low-exposure behavior that performed better than the buy-collapsed regularized
DDQN, but worse than cash-preserving Q-Learning and slightly worse than the
best unregularized DDQN run.

The trader-archetype notebook adds interpretability rather than another
predictive model. All four notebook smoke tests passed, and the Shapley
rankings produced coherent stories for the three hand-crafted trader types.
Short-horizon returns dominate the rational arbitrageur, volume dominates the
manipulator, and volume plus recent momentum dominate herd-following retail.

Taken together, these results reinforce the main lesson of the broader project:
the hardest part is not building another algorithm. It is extracting stable,
generalizable decision rules from a market that is short-lived, noisy, and
structurally unstable. REINFORCE changes the training dynamics, and the Shapley
notebook improves interpretability, but neither removes the basic difficulty of
learning rational behavior in a speculative meme-coin market.
