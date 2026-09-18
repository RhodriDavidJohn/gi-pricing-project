"""Load, clean and join the freMTPL2 frequency and severity data.

Feature pre-processing follows the GLM case study on this data (Noll, Salzmann & Wüthrich,
"Case Study: French Motor Third-Party Liability Claims", 2018): claim counts capped at 4 and
exposure at 1 (larger values are treated as data errors), driver age, vehicle age and vehicle
power grouped, bonus-malus capped at 150, log density, Area as an ordered code, and Region /
VehBrand as categories.
"""

import json
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FREQ_FILE = "freMTPL2freq.csv"
SEV_FILE = "freMTPL2sev.csv"
POLICY_FILE = "clean_policy_data.csv"
FEATURE_INFO_FILE = "feature_info.json"

# the customer details needed to price a policy (raw freMTPL2freq columns)
INPUT_COLUMNS = [
    "DrivAge", "BonusMalus", "VehAge", "VehPower", "VehBrand", "VehGas", "Density", "Area", "Region",
]
CATEGORICAL_INPUTS = ["VehBrand", "VehGas", "Area", "Region"]

# severity models are fitted to claim amounts capped at the large claim threshold; the rebase
# factor (set on uncapped claims cost) spreads the cost above the cap across all policies
SEVERITY_TARGET = "MeanClaimAmountCapped"
LOG_SEVERITY_TARGET = "LogClaimAmountCapped"


def load_raw_data(raw_dir):
    df_frq = pd.read_csv(raw_dir / FREQ_FILE)
    df_sev = pd.read_csv(raw_dir / SEV_FILE)

    if df_frq["IDpol"].duplicated().any():
        raise ValueError("IDpol is not unique in the frequency data")

    logger.info("Loaded %d policies and %d claims", len(df_frq), len(df_sev))
    return df_frq, df_sev


# ----------------------------------------------------
# Features
# ----------------------------------------------------
def band(values, edges):
    """Group values into [edge, next edge) bands labelled like "18-20" and "71+".
    Values below the first edge go into the first band."""
    edges = list(edges)
    labels = [f"{lo:,.0f}-{hi - 1:,.0f}" if hi - lo > 1 else f"{lo:,.0f}" for lo, hi in zip(edges, edges[1:])]
    labels.append(f"{edges[-1]:,.0f}+")
    banded = pd.cut(values.clip(lower=edges[0]), [*edges, np.inf], labels=labels, right=False)
    return banded.astype(str)


def band_labels(edges):
    return band(pd.Series(edges, dtype=float), edges).tolist()


def clean_text(values):
    # raw text values can come quoted, e.g. "'Regular'"
    return values.astype(str).str.replace("'", "")


def feature_info(df_frq_raw, df_sev_raw, cfg):
    """Everything the features need from the training data, so new data is encoded the same way:
    the levels of each categorical feature and the large claim cap. `cfg` is the `features` section."""
    levels = {col: sorted(clean_text(df_frq_raw[col]).unique().tolist()) for col in CATEGORICAL_INPUTS}
    for col, edges in cfg["bands"].items():
        levels[f"{col}Band"] = band_labels(edges)

    percentile = cfg.get("large_claim_percentile")
    claim_cap = None if percentile is None else float(np.percentile(df_sev_raw["ClaimAmount"], percentile))

    return {"levels": levels, "claim_cap": claim_cap}


def set_categories(df, levels):
    """Categorical columns with a fixed set of levels (unknown values become missing)."""
    for col, categories in levels.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col].astype(str), categories=categories)
    return df


def clean_frequency_data(df_frq, cfg, info):
    """Add the rating factors to the policy data.

    cfg: the `features` config section. info: output of feature_info (levels, claim cap).
    """
    df = df_frq.copy()

    # caps: larger claim counts and exposures are treated as data errors
    for col, cap in cfg["caps"].items():
        if col in df.columns:
            df[col] = df[col].clip(upper=cap)

    # a few days of cover gives a wild claim rate (one claim on 0.003 years is 300 a year),
    # so those policies are dropped
    min_exposure = cfg.get("min_exposure")
    if min_exposure and "Exposure" in df.columns:
        keep = df["Exposure"] >= min_exposure
        if not keep.all():
            logger.info(
                "Dropped %d policies (%.2f%% of policies, %.2f%% of exposure) below %s years of exposure",
                (~keep).sum(), 100 * (~keep).mean(),
                100 * df.loc[~keep, "Exposure"].sum() / df["Exposure"].sum(), min_exposure,
            )
        df = df[keep].copy()

    for col in CATEGORICAL_INPUTS:
        df[col] = clean_text(df[col])

    # density is heavily skewed, so it is used on the log scale
    df["LogDensity"] = np.log(df["Density"].clip(lower=1e-6))

    # young driver and new vehicle flags, used in the candidate interactions
    df["YoungDrivAge"] = np.where(df["DrivAge"] <= cfg["young_driver_max_age"], 1, 0)
    df["NewVehAge"] = np.where(df["VehAge"] <= cfg["new_vehicle_max_age"], 1, 0)

    # ---------------- rating factors from the case study ----------------
    # bonus-malus above the cap is rare, so its effect is held flat beyond it
    df["BonusMalusCapped"] = df["BonusMalus"].clip(upper=cfg["bonus_malus_cap"])

    # driver age, vehicle age and vehicle power in groups (e.g. DrivAgeBand "18-20", "71+")
    for col, edges in cfg["bands"].items():
        df[f"{col}Band"] = band(df[col], edges)

    # Area A-F follows density, so it is used as an ordered code 1-6
    df["AreaCode"] = df["Area"].map({area: i + 1 for i, area in enumerate(info["levels"]["Area"])})

    df = set_categories(df, info["levels"])

    # identify policies with >0 claims (not available when pricing new customers)
    if "ClaimNb" in df.columns:
        df["PosClaims"] = np.where(df["ClaimNb"] > 0, 1, 0)

    return df.drop(columns=["Density"])


def build_policy_data(df_frq, df_sev, claim_cap=None):
    """Join per-policy claim amounts onto the cleaned policy data (one row per policy)."""
    df_sev = df_sev.assign(
        ClaimAmountCapped=df_sev["ClaimAmount"] if claim_cap is None else df_sev["ClaimAmount"].clip(upper=claim_cap)
    )

    # mean, total and number of claim amounts per policy
    df_claims = df_sev.groupby("IDpol", as_index=False).agg(
        MeanClaimAmount=("ClaimAmount", "mean"),
        MeanClaimAmountCapped=("ClaimAmountCapped", "mean"),
        TotalClaimAmount=("ClaimAmount", "sum"),
        NbSevClaims=("ClaimAmount", "size"),
    )

    # the log-normal severity model is fitted to the log of the capped mean
    df_claims["LogClaimAmountCapped"] = np.log(df_claims["MeanClaimAmountCapped"].clip(lower=1e-6))

    df = pd.merge(left=df_frq, right=df_claims, on="IDpol", how="left", validate="one_to_one")

    # add a flag for severity rows
    df["sev_flag"] = np.where(df["MeanClaimAmount"].notna(), 1, 0)

    # policies without claim amounts cost nothing
    df["TotalClaimAmount"] = df["TotalClaimAmount"].fillna(0)
    df["NbSevClaims"] = df["NbSevClaims"].fillna(0).astype(int)

    return df


def check_claim_counts(df):
    """Log where the two raw files disagree: frequency is fitted to ClaimNb and severity to the
    claim amounts, so a policy can have a claim with no amount, or an amount with no claim."""
    if "ClaimNb" not in df.columns:
        return

    no_amount = (df["ClaimNb"] > 0) & (df["NbSevClaims"] == 0)
    no_count = (df["ClaimNb"] == 0) & (df["NbSevClaims"] > 0)
    # ClaimNb is capped before this, so only differences below the cap are counted here
    differs = (df["ClaimNb"] > 0) & (df["NbSevClaims"] > 0) & (df["ClaimNb"] != df["NbSevClaims"])

    logger.info(
        "Claim counts vs claim amounts: %d policies with claims but no amounts, %d with amounts but "
        "no claims, %d where the counts differ (%.2f%% of policies with either)",
        no_amount.sum(), no_count.sum(), differs.sum(),
        100 * (no_amount | no_count | differs).sum() / max((df["ClaimNb"] > 0).sum() + no_count.sum(), 1),
    )


def build_features(df_frq_raw, df_sev_raw, cfg):
    """Policy table with features, and the feature info used to build it. `cfg` is the full config."""
    info = feature_info(df_frq_raw, df_sev_raw, cfg["features"])
    df = build_policy_data(
        clean_frequency_data(df_frq_raw, cfg["features"], info), df_sev_raw, info["claim_cap"]
    )
    if info["claim_cap"] is not None:
        capped_share = float((df_sev_raw["ClaimAmount"] > info["claim_cap"]).mean())
        logger.info(f"Large claim cap: {info['claim_cap']:,.0f} ({capped_share:.2%} of claims above it)")
    check_claim_counts(df)
    return df, info


def prepare_data(cfg):
    df_frq, df_sev = load_raw_data(cfg["paths"]["raw_dir"])
    df, info = build_features(df_frq, df_sev, cfg)

    processed_dir = cfg["paths"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_dir / POLICY_FILE, index=False)
    (processed_dir / FEATURE_INFO_FILE).write_text(json.dumps(info, indent=2))
    logger.info("Saved cleaned data to %s", processed_dir / POLICY_FILE)

    return df


def load_policy_data(cfg):
    processed_dir = cfg["paths"]["processed_dir"]
    if not (processed_dir / POLICY_FILE).exists():
        raise FileNotFoundError(f"{processed_dir / POLICY_FILE} not found - run the 'prepare' step first")

    info = json.loads((processed_dir / FEATURE_INFO_FILE).read_text())
    return set_categories(pd.read_csv(processed_dir / POLICY_FILE), info["levels"])
