# feature_selection.py

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Mesmas colunas de leakage removidas no preprocessing.py
LEAK_COLUMNS = [
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

def apply_leak_filter(df):
    """Remove as mesmas colunas de leakage usadas no preprocessing."""
    cols = [c for c in df.columns if c not in LEAK_COLUMNS]
    return df[cols]

def select_features(X, y, n_features=10):
    """
    Seleciona features com base na importância calculada pela DecisionTree.
    Agora usando o mesmo filtro anti-leak do preprocessing.
    """

    # Aplicar o mesmo anti-leak
    X_filtered = apply_leak_filter(X)

    # Modelo simples apenas para medir importâncias
    model = DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_filtered, y)

    importance = pd.DataFrame({
        "feature": X_filtered.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\n=== IMPORTÂNCIA DAS FEATURES ===")
    print(importance)

    # Seleciona as N mais importantes
    selected = list(importance.head(n_features)["feature"])

    print("\nTotal selecionadas:", len(selected))
    print("Features selecionadas:", selected)

    return selected
