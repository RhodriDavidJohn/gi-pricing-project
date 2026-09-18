from sklearn.model_selection import train_test_split


def train_val_test_df_split(df, stratify_col=None, val_size=0.2, test_size=0.2, random_state=42):
    """Split a DataFrame into train, validation and test sets."""
    val_test_size = val_size + test_size

    df_train, df_val_test = train_test_split(
        df,
        stratify=df[stratify_col] if stratify_col else None,
        test_size=val_test_size,
        random_state=random_state,
    )
    df_val, df_test = train_test_split(
        df_val_test,
        stratify=df_val_test[stratify_col] if stratify_col else None,
        test_size=test_size / val_test_size,
        random_state=random_state,
    )

    return df_train, df_val, df_test


def split_policy_data(df, cfg):
    """The split shared by the GLM and GBM scripts, so their test results are comparable."""
    return train_val_test_df_split(
        df,
        stratify_col="PosClaims",
        val_size=cfg["split"]["val_size"],
        test_size=cfg["split"]["test_size"],
        random_state=cfg["random_state"],
    )


def severity_rows(df):
    """Policies with at least one claim amount."""
    return df[df["sev_flag"] == 1]
