from sklearn.preprocessing import LabelEncoder

def clean_target(df):
    df["delivery_status"] = df["delivery_status"].astype(str).str.upper().str.strip()

    map_status = {
        "DELIVERED": 1,
        "CANCELLED": 0,
        "CANCELED": 0,
        "FAILED": 0,
        "RETURNED": 0,
    }

    df["target"] = df["delivery_status"].map(map_status)
    df = df.dropna(subset=["target"])
    print("Shape após limpar target:", df.shape)
    return df

def remove_leakage(df):
    cols_leak = [
        "order_status",
        "payment_status",
        "order_moment_collected",
        "order_moment_in_expedition",
        "order_moment_delivering",
        "order_moment_delivered",
        "order_moment_finished",
        "order_metric_collected_time",
        "order_metric_paused_time",
        "order_metric_production_time",
        "order_metric_walking_time",
        "order_metric_expediton_speed_time",
        "order_metric_transit_time",
        "order_metric_cycle_time",
    ]

    for col in cols_leak:
        if col in df.columns:
            df = df.drop(columns=col)
    return df

def handle_missing_values(df):

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())


    cat_cols = df.select_dtypes(include=["object"]).columns
    for c in cat_cols:
        df[c] = df[c].fillna("UNKNOWN")
    return df

def encode_categoricals(X):
    cat_cols = X.select_dtypes(include=["object"]).columns
    le = LabelEncoder()
    for c in cat_cols:
        X[c] = le.fit_transform(X[c].astype(str))
    return X
