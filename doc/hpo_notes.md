# HPO notes

Practical notes on hyperparameter tuning in this repo — what's tuned, what's been intentionally frozen, and how to extend the search space.

## The search space (`WeeklyRotationRunner.dimensions`)

Active dimensions, in `exp/strategy/weekly_rotation.py`:

| Name | Range | Prior | Notes |
|---|---|---|---|
| `lookback` | `Real(50, max_lookback)` | log-uniform | SMA & ROC lookback. Bounded above by `max_lookback` to keep the precomputed window valid. Cast to `int` in the runner. |
| `rsi_lookback` | `Real(3, 100)` | log-uniform | RSI window. Cast to `int`. |
| `rsi_threshold` | `Real(47.0, 53.0)` | uniform | Narrowly bracketed around the rulebook's 50 — see "tight ranges" below. |
| `day_of_trade` | `Categorical([0,1,2,3,4], onehot)` | — | Weekday to rebalance. The rulebook says Friday (4); we let HPO check the others. |

Frozen (commented-out) dimensions still in the file:

| Name | Default | Why frozen |
|---|---|---|
| `sma_tol` | 0.02 | SPY-above-SMA filter tolerance. Tuning it lets HPO disable the regime filter, which we don't want — the filter is the whole point of the rule set. |
| `volume_lookback`, `volume_threshold` | 20, 1e6 | Liquidity floor. Tuning either lets HPO trade illiquid names whose fill prices the backtester can't model accurately. |
| `price_min` | 1.0 | Penny-stock floor. Same fill-realism argument; also feeds into `metrics.get_notional` so changing it shifts position sizing. |
| `n_positions` | 10 | Portfolio breadth. Holding fewer names dramatically increases variance and makes Sharpe/Sortino noisier — better held fixed while tuning signal parameters. |

If you uncomment any of these, also add an entry to the runner's `signs` dict if it's a new *metric*. The frozen *parameters* don't need that, but they do need a sensible default kept in `WeelkyRotationStrategy.__init__`.

## Why some ranges are deliberately tight

`rsi_threshold ∈ [47, 53]` looks restrictive but is intentional. The strategy is parameterized around the published rule set, and the goal of HPO here is **regularization-style fine-tuning** around the known-good defaults, not free-form search. Wider ranges produce more profitable in-sample results that don't survive walk-forward validation. The same logic explains why `lookback` is bounded below at 50 rather than 1 — anything shorter is a different strategy, not a tuned version of this one.

When adding a new dimension, default to the narrowest range that still spans "plausibly correct" values, then widen only if val-set metrics improve along with train-set metrics.

## Optimizer choice

Set via `optimizer=` kwarg on `train_strategy` / `cross_validate_strategy` / `cv_opt_driver`. Three options registered in `OPTIMIZER_FUNCTION` / `OPTIMIZER_KWARGS`:

| Key | skopt function | Surrogate | Acquisition |
|---|---|---|---|
| `'forest'` | `skopt.forest_minimize` | Extra Trees | LCB |
| `'GBRT'` (default) | `skopt.gbrt_minimize` | Gradient Boosted Trees | LCB |
| `'GP'` | `skopt.gp_minimize` | Gaussian Process | GP-hedge |

GBRT is the default because it's the most robust to the mixed Real + Categorical search space here (GP needs careful kernel design when categoricals are present, and forest is more sample-hungry). Use GP only for purely-Real spaces with smooth objectives.

## Acquisition kwargs (in `OPTIMIZER_KWARGS`)

Passed through to skopt. Two knobs to be aware of:
- `xi=0.01`, `kappa=1.96` — set in `train_strategy`, control exploration vs exploitation. Defaults are fine for ~50–200 calls per fold. Lower `xi`/`kappa` for tighter exploitation if the search space is already narrow.
- `n_points=10000` — number of candidate points the acquisition function is evaluated on per iteration. Keep this large because the search space is low-dimensional, so the cost is negligible.

## Tuning the tuning: how many calls?

A useful working point on this strategy:

```
n_folds        = 4
train_window   = 520w  (~10 years)
val_window     = 26w   (~6 months)
n_iters        = 20    # outer loops (each adds n_calls per fold)
n_calls        = 10    # per iteration per fold
n_rand         = 10    # only on the very first iteration; afterwards = 0
```

This gives 4 folds × 20 iters × 10 calls = 800 evaluations per fold, with the first 10 random and the remaining 790 model-guided. Convergence plots in `results/run_*/fold_NN.png` will plateau well before that for the current 4-dim active search space — drop `n_iters` if you see flatlining.

## Sign convention

skopt minimizes. Every metric in `WeeklyRotationRunner.signs` maps to `-1`:

```python
signs = {PL: -1, YIELD: -1, SHARPE: -1, SORTINO: -1}
```

If you add a metric where larger is *worse* (e.g. max drawdown), use `+1`. The runner multiplies `signs[output_metric] * metric` inside `skopt_func`.

## Resume

Two resume mechanisms compose (see [ARCHITECTURE.md](ARCHITECTURE.md) §"Resume semantics"). For HPO specifically: each fold's `checkpoint_<dates>_<metric>.pkl` is a full `skopt.OptimizeResult` and can be loaded directly with `skopt.load(path)` to inspect `x_iters`, `func_vals`, and the model's surrogate. Useful for hand-debugging a fold that looks suspicious.

## Pitfalls

- **`max_lookback` is structural, not a hyperparameter.** It sets how much history is loaded before `start_date_requested`. Bump it if you add a longer-period indicator; otherwise leave at 200.
- **Verbose + parallel is forbidden** (`cv_opt_driver` asserts `n_jobs == 1` when `verbose=True`). joblib workers can't share the logger.
- **First iteration uses `n_rand`, subsequent ones don't.** `cross_validate_strategy` overrides `n_rand=0` for iterations > 0. This is correct (you don't want to re-seed randomly on resume) but easy to miss when reading the code.
- **The HPO objective is computed on the train window only.** Val metrics are recorded but don't drive optimization. If you want to penalize train/val divergence directly, build a composite metric in `make_backtesting_report` and feed it through `output_metric`.
