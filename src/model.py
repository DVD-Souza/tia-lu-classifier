from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print("\n=== TESTE ===")
    print("Accuracy:", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred))
    print("Recall:", recall_score(y_test, pred))
    print("F1:", f1_score(y_test, pred))

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, pred))

    print("\n=== CONFUSION MATRIX ===")
    print(confusion_matrix(y_test, pred))

    print("\nAUC:", roc_auc_score(y_test, pred))

    print("\n=== CROSS VALIDATION (5 folds) ===")
    scores = cross_val_score(model, X, y, cv=5, scoring="f1")
    print("F1 mean:", scores.mean())
    print("F1 std:", scores.std())
