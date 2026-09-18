"""GLM frequency and severity models (statsmodels).

Models are specified with feature names (e.g. "DrivAgeBand", "DrivAgeBand:BonusMalusCapped").
The fit functions turn these into patsy terms - categorical features become
C(feature, Treatment(reference=<level with the most exposure>)) - and store the expanded terms on
the results as `results.terms`, so predictions always use the same coding.
"""

import logging
import warnings

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import HessianInversionWarning

from lib.data import LOG_SEVERITY_TARGET, SEVERITY_TARGET
from lib.split import severity_rows

logger = logging.getLogger(__name__)

BASE_MODEL = "(none)"
MIN_NB_ALPHA = 1e-8


def build_formula(response, terms):
    return f"{response} ~ {' + '.join(terms)}"


def expand_terms(features, df):
    """Feature names -> patsy terms. Categorical features use the level with the most exposure
    as the base level (the usual choice for rating factor relativities)."""
    def patsy_name(feature):
        if isinstance(df[feature].dtype, pd.CategoricalDtype):
            base = df.groupby(feature, observed=True)["Exposure"].sum().idxmax()
            return f"C({feature}, Treatment(reference={str(base)!r}))"
        return feature

    return [":".join(patsy_name(part) for part in term.split(":")) for term in features]


def design_matrix(terms, df):
    """Right-hand side design matrix only.

    patsy.dmatrices drops rows where the response is NaN, which silently removes
    every non-claiming policy when the response is a claim amount.
    """
    return patsy.dmatrix(" + ".join(terms), df, return_type="dataframe")


def linear_predictor(params, terms, df):
    """X * beta using saved coefficients only (no statsmodels results needed)."""
    X = design_matrix(terms, df)
    return X[params.index].to_numpy() @ params.to_numpy()


# ----------------------------------------------------
# Frequency
# ----------------------------------------------------
def fit_frequency_glm(df, terms, family="poisson", nb_alpha=1.0):
    """Claim count GLM with log(Exposure) offset."""
    if family == "poisson":
        glm_family = sm.families.Poisson()
    elif family == "negative_binomial":
        glm_family = sm.families.NegativeBinomial(alpha=nb_alpha)
    else:
        raise ValueError(f"Unknown frequency family: {family}")

    terms = expand_terms(terms, df)
    y, X = patsy.dmatrices(build_formula("ClaimNb", terms), data=df, return_type="dataframe")

    results = sm.GLM(endog=y, exog=X, family=glm_family, exposure=df["Exposure"]).fit()
    results.terms = terms
    return results


def estimate_nb_alpha(df, terms):
    """Maximum likelihood estimate of the negative binomial dispersion (NB2: Var = mu + alpha * mu^2).

    The GLM negative binomial family needs alpha fixed, so it is estimated here with the
    full negative binomial likelihood, starting from the Poisson coefficients.
    """
    y, X = patsy.dmatrices(
        build_formula("ClaimNb", expand_terms(terms, df)), data=df, return_type="dataframe"
    )
    poisson_results = sm.GLM(
        endog=y, exog=X, family=sm.families.Poisson(), exposure=df["Exposure"]
    ).fit()

    nb_model = sm.NegativeBinomial(y, X, exposure=df["Exposure"], loglike_method="nb2")
    start_params = np.append(poisson_results.params, 0.1)

    # Newton's method first: statsmodels' default (BFGS) can stop at the starting alpha when
    # the likelihood is flat in alpha. Only a converged, positive alpha is accepted.
    for method in ["newton", "bfgs"]:
        # near alpha = 0 statsmodels warns about the Hessian - convergence is checked below instead
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", HessianInversionWarning)
            nb_results = nb_model.fit(start_params=start_params, method=method, maxiter=200, disp=0)
        alpha = float(nb_results.params["alpha"])
        if nb_results.mle_retvals["converged"] and np.isfinite(alpha) and alpha > 0:
            logger.info("Estimated negative binomial alpha: %.4f", alpha)
            return alpha

    # no usable estimate: the data shows no clear overdispersion, so use (almost) Poisson
    logger.warning("Negative binomial alpha could not be estimated - using %g", MIN_NB_ALPHA)
    return MIN_NB_ALPHA


def predict_frequency(results, df):
    """Annual claim frequency.

    statsmodels predict() with new data uses exposure = 1, so this is already a
    rate per policy year - multiply by Exposure to get expected claims.
    """
    return np.asarray(results.predict(design_matrix(results.terms, df)))


# ----------------------------------------------------
# Severity
# ----------------------------------------------------
def fit_gamma_glm(df, terms):
    """Gamma GLM (log link) on the (capped) mean claim amount, weighted by number of claims."""
    df = severity_rows(df)
    terms = expand_terms(terms, df)
    y, X = patsy.dmatrices(
        build_formula(SEVERITY_TARGET, terms), data=df, return_type="dataframe"
    )

    results = sm.GLM(
        endog=y,
        exog=X,
        family=sm.families.Gamma(link=sm.families.links.Log()),
        var_weights=df["NbSevClaims"],
    ).fit()
    results.terms = terms
    return results


def predict_gamma_severity(results, df):
    return np.asarray(results.predict(design_matrix(results.terms, df)))


def fit_log_normal_wls(df, terms):
    """Weighted least squares on the log (capped) mean claim amount."""
    df = severity_rows(df)
    terms = expand_terms(terms, df)
    y, X = patsy.dmatrices(
        build_formula(LOG_SEVERITY_TARGET, terms), data=df, return_type="dataframe"
    )

    results = sm.WLS(endog=y, exog=X, weights=df["NbSevClaims"]).fit()
    results.terms = terms
    return results


def smearing_factor(results):
    """Duan's smearing factor, weighted the same way as the model."""
    return float(np.average(np.exp(results.resid), weights=results.model.weights))


def predict_log_normal_severity(results, df):
    # just using exp(pred) under-predicts the true mean due to Jensen's inequality
    log_predictions = np.asarray(results.predict(design_matrix(results.terms, df)))
    return np.exp(log_predictions) * smearing_factor(results)


# ----------------------------------------------------
# Interaction selection
# ----------------------------------------------------
def information_criterion(results, criterion="aic"):
    if criterion == "aic":
        return results.aic
    if criterion == "bic":
        # GLM results call the likelihood-based BIC bic_llf
        return getattr(results, "bic_llf", results.bic)
    raise ValueError(f"Unknown criterion: {criterion}")


def select_interactions(fit_fn, df, covariates, candidates, min_improvement=2.0, criterion="aic",
                        **fit_kwargs):
    """Forward selection of interactions by AIC or BIC.

    Starting from the covariates, repeatedly add the candidate that lowers the criterion the most,
    and stop when the best one lowers it by less than `min_improvement`. Both criteria penalise
    each extra parameter (BIC more heavily on large data), so the base model can win.

    Returns the selected terms (feature names) and a table of the steps taken.
    """
    def score(terms):
        return information_criterion(fit_fn(df, terms, **fit_kwargs), criterion)

    terms = list(covariates)
    remaining = list(candidates)
    current = score(terms)
    history = [{"step": 0, "added": BASE_MODEL, criterion: current, "improvement": None}]

    while remaining:
        candidate_scores = {c: score(terms + [c]) for c in remaining}
        best = min(candidate_scores, key=candidate_scores.get)
        improvement = current - candidate_scores[best]

        if improvement < min_improvement:
            break

        terms.append(best)
        remaining.remove(best)
        current = candidate_scores[best]
        history.append({"step": len(history), "added": best, criterion: current, "improvement": improvement})

    logger.info("Selected interactions (%s): %s", criterion.upper(), terms[len(covariates):] or BASE_MODEL)
    return terms, pd.DataFrame(history)
