# Development notes

How this project got from the original notebook (`notebooks/full_workflow_original.ipynb`) to the current scripts.

## Bugs fixed

| # | Notebook | Problem | Fix |
|---|---|---|---|
| 1 | cell 30 | `patsy.dmatrices(gamma_formula, df_frq_test)` drops every policy with no claim (NaN response), so most premiums were NaN (premium $791k, Gini -0.01, NaN tariff) | Predictions use `patsy.dmatrix` on the right-hand side only (`glm.design_matrix`) |
| 2 | cell 25 | Log-normal model fitted with the leftover `formula` from cell 22, not `ln_formula` | Each model is fitted from its own terms inside a function, so there is no shared `formula` variable |
| 3 | cells 24, 27 | Pure premium used `VehAge:BonusMalus` for Gamma although the selection picked a different interaction | Pure premium uses the Gamma terms chosen by the selection step |
| 4 | cells 16, 27 | Frequency model evaluated (two interactions) differed from the one priced (one interaction) | One frequency formula (chosen by the selection step) for both |
| 5 | cell 13 | Predicted annual rate divided by exposure again; MSE compared annual rates with claim counts; Gini sorted on rate ÷ exposure | `predict_frequency` returns the annual rate; counts = rate × exposure |
| 6 | cells 32, 33, 36 | Actual cost was `MeanClaimAmount`, understating multi-claim policies | `TotalClaimAmount` (sum of claim amounts per policy) |
| 7 | cell 33 | Rebase factor calculated on the test set, so the test loss ratio was 1 by construction | Calculated on the validation set, applied to test |
| 8 | cells 9, 41 | GLM and GBM used different test sets | One train/val/test split shared by both |
| 9 | cell 52 | Hurdle predictions compared with `PosClaims` (0/1) instead of `ClaimNb` | Compared with `ClaimNb` |
| 10 | cells 21, 22 | The Gamma and log-normal loops tested different interaction lists | One list in config (`glm.interactions_to_test`) |
| 11 | cells 21, 22, 25 | Severity weights used `ClaimNb`, which doesn't always match the number of claim amounts averaged | Weighted by `NbSevClaims` (number of rows in the severity file) |
| 12 | cell 19 | Smearing factor unweighted for a weighted model | Weighted by the model weights |
| 13 | cell 4 | `pd.cut` bins were `(0, 300]` etc., contradicting the comment (`<300`, `[300, 1500)`) | `right=False` |
| 14 | cells 13, 32 | Lorenz curve didn't start at (0, 0), slightly biasing the Gini | Curve starts at the origin |
| 15 | cells 32–36 | NaN premiums were skipped silently by pandas sums | `evaluate_pure_premium` raises if any premium is NaN |

Unused code was removed: `X, y` (cell 9), the `DMatrix` objects (cell 51) and the `IDpol`
fallback in the tariff printout.

## Method improvements

**Interaction selection (`glm.select_interactions`)**
- Forward selection by AIC for the Poisson frequency, Gamma and log-normal models: add the
  candidate that lowers AIC most, and stop when the best one lowers it by less than
  `glm.min_aic_improvement` (default 2; now `glm.min_improvement`). The notebook picked the highest log-likelihood,
  which almost always adds an interaction. AIC charges for the extra parameter, so the model
  can end up with no interactions, or with more than one.
- The negative binomial model uses the same terms as the Poisson model, so the two are compared fairly.
- Each step is saved to `outputs/selection/glm/interaction_selection_<model>.csv`.
- The Gamma AIC uses statsmodels' estimated dispersion, so treat small AIC differences with care.

**Negative binomial alpha (`glm.estimate_nb_alpha`)**
- `alpha` is estimated by maximum likelihood (NB2, `Var = mu + alpha * mu^2`) with the same
  exposure offset, then used in the negative binomial GLM. It's saved as `frequency.nb_alpha`
  in `metrics.json`. Set `glm.nb_alpha` to a number to fix it instead.
- Newton's method is tried first (BFGS can stop at the starting value when the likelihood is
  flat in alpha). If no positive alpha converges, a near-zero alpha (effectively Poisson) is used.
- An alpha near 0 means little overdispersion, so the negative binomial and Poisson results will
  be almost the same.

**Minimum premium (`pricing.minimum_premium_from_percentile`)**
- Annual premiums are floored at the `pricing.minimum_premium_percentile` percentile
  (default 5) of the rebased validation premiums (holdout premiums for the final models).
  Set it to `null` to switch the floor off.
- The floor is set on the validation set, like the rebase factor, so the test set only reports
  results.
- `metrics.json` gives the floor (`minimum_premium`) and the share of test policies raised to it
  (`share_raised_to_minimum`).
- `annual_tariff` shows the effect on loss ratio. Raising low premiums adds premium, so the
  loss ratio drops slightly below the rebased value.

**XGBoost hurdle model (`gbm.py`)**
- Exposure is now a feature in both parts instead of a log(Exposure) offset on the positive
  part. An offset there scaled "claims given a claim" below 1 for short policies, which is
  impossible. The classifier previously ignored exposure, although the chance of a claim
  depends heavily on it.
- The positive part models `ClaimNb - 1` (Poisson), so predicted claims given a claim are
  always at least 1.
- Both parts are constrained so that predictions never go down as exposure goes up.
- Annual frequency is predicted with Exposure = 1, instead of dividing by exposure.

**Simple XGBoost frequency model (`gbm.fit_simple_frequency`)**
- One Poisson model on `ClaimNb` with a log(Exposure) offset, as a benchmark for the hurdle
  model. The offset is exact here because the model is for the total claim count.
- The offset also contains the portfolio's average claim rate: XGBoost ignores its own starting
  value when an offset is given, so without it the trees start from one claim a year.

## Rating factor changes

**Features** (see the Rating factors section of the README):
- Claim counts are capped at 4 and exposure at 1.
- The GLM uses grouped driver age, vehicle age and vehicle power, following Noll, Salzmann &
  Wüthrich (2018). Before this it used yes/no flags such as young driver.
- Bonus-malus is capped at 150. Density is logged. Area is an ordered code. Brand, fuel and
  region are categories; region wasn't used before.
- XGBoost now uses raw values and categories rather than the flags, and its frequency models
  can only increase with bonus-malus.
- Severity models are fitted to claim amounts capped at the 99.5th percentile. The cost above
  the cap reaches the premium through the rebase factor.

**GLM:**
- Categorical terms use the level with the most exposure as the base.
- Model terms are stored on the fitted results, so predictions always use the same coding.
- Interactions are chosen by BIC by default. With hundreds of thousands of policies, AIC keeps
  almost any multi-level interaction. The candidate list was rewritten, with a reason for each.

**Explanations:**
- `explain` now computes exact Shapley values against a reference customer by quoting every mix
  of the two customers' details, so it works for XGBoost as well as the GLM.
- It replaces the earlier coefficient-based version, which only handled numeric GLM terms and
  split interaction effects approximately.

**Encoding:** category levels and the claim cap are saved (`feature_info.json`, and inside
each final model), so single quotes are encoded like the training data.

## Streamlit app

- Built on the pipeline outputs only: the app never refits a model, so it starts quickly and
  shows exactly what was selected and trained.
- Quotes, explanations and price curves are cached by customer, so moving a slider back to an
  earlier value is instant.
- The GLM is the default quote model because its calculation can be shown term by term
  (`GLMPricingModel.calculation`). XGBoost quotes are explained with the same Shapley method.
- `report.py` now also saves the Lorenz curves as a table (`reports/tables/lorenz.csv`) for the
  app's chart.
- The candidate interactions in `config.yaml` now carry their reason, shown on the model
  selection page.
- `.gitignore` now allows the selection outputs to be committed (apart from the large
  per-policy files) so the app can be deployed without the data.

## Cleaning tidy-up

- The features left over from the original notebook (`VehGasRegular`, `DensityCat`,
  `DrivAgeCubed`, `OldDrivAge`, `OldVehAge`, `HighVehPower`, `GoodDrivingBonus`, `VehBrandB12`)
  were removed: the banded rating factors replaced them, and nothing read them. Their config
  thresholds went with them. `YoungDrivAge` and `NewVehAge` are kept because the candidate
  interactions use them.
- `LogClaimAmount` (uncapped) was removed too; the log-normal severity model uses
  `LogClaimAmountCapped`.
- **Very short policies are dropped** (`features.min_exposure`, 0.02 years - about a week). A
  claim on a few days of cover implies a claim rate in the hundreds, which the frequency models
  chase. The number removed is logged and reported in the results summary.
- **Claim counts vs claim amounts** are now compared after the join, and the disagreements are
  logged: policies with claims but no amounts, amounts but no claims, and differing counts.
