import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from utils import get_config, parse_args, save_model
from data_processing import process_data_for_network_activity_classification

def fit_evaluate_model(data_processed):
    """
    Train and evaluate a gradient boosting classifier using stratified 5 folds cross validation.

    Parameters
    ----------
    data_processed : pd.DataFrame
        Processed dataset

    Returns
    -------
    model: GradientBoostingClassifier object
        Trained model
    """
    df_X = data_processed.drop(columns=["target"])
    df_y = data_processed["target"]
    X, y = df_X.values, df_y.values

    param_grid = {
        "n_estimators": [10 , 50, 100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5]}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=GradientBoostingClassifier(random_state=42),
        param_grid=param_grid,
        scoring='f1',
        cv=skf)

    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")

    f1_score_all_folds = []
    precision_score_all_folds = []
    accuracy_score_all_folds = []
    recall_score_all_folds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        best_model.fit(X_train, y_train)
        y_proba = best_model.predict_proba(X_val)[:, 1]

        f1_scores = []
        precision_scores = []
        recall_scores = []
        accuracy_scores = []

        for proba_threshold in np.arange(0, 1, 0.01):
            y_pred = (y_proba > proba_threshold).astype(int)
            f1 = f1_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred)
            accuracy = accuracy_score(y_val, y_pred)

            f1_scores.append(f1)
            precision_scores.append(precision)
            recall_scores.append(recall)
            accuracy_scores.append(accuracy)

        f1_score_all_folds.append(f1_scores)
        precision_score_all_folds.append(precision_scores)
        recall_score_all_folds.append(recall_scores)
        accuracy_score_all_folds.append(accuracy_scores)

    average_f1 = np.mean(np.array(f1_score_all_folds), axis=0)
    average_precision = np.mean(np.array(precision_score_all_folds), axis=0)
    average_accuracy = np.mean(np.array(accuracy_score_all_folds), axis=0)
    average_recall = np.mean(np.array(recall_score_all_folds), axis=0)

    best_model.fit(X, y)

    fig, ax = plt.subplots(figsize=(15,6))
    ax.plot(np.arange(0, 1, 0.01), average_f1, label="F1score")
    ax.plot(np.arange(0, 1, 0.01), average_precision, label="Precision")
    ax.plot(np.arange(0, 1, 0.01), average_accuracy, label="Accuracy")
    ax.plot(np.arange(0, 1, 0.01), average_recall, label="Recall")

    best_threshold = np.arange(0, 1, 0.01)[np.argmax(average_f1)]
    ax.axvline(x=best_threshold, color="k",
               label=f"Best threshold for F1: {best_threshold:.2f}", linestyle="--")
    ax.text(best_threshold, -0.03, f"{best_threshold:.2f}", color='k', fontsize=9)

    ax.set_title("Classification metrics for gradient boosting classifier after grid search on 5 folds")
    ax.set_xlabel("Prediction probability threshold")
    ax.set_ylabel("Metric value")
    ax.legend()
    fig.savefig("./models/results/network_activity_classifier_metrics.png")

    return best_model


def main(args):
    """
    main function.
    """
    config = get_config(args.configuration)
    df_processed = process_data_for_network_activity_classification(config)
    model = fit_evaluate_model(df_processed)
    save_model(model, "./models/network_activity_classifier.pkl")



if __name__ == "__main__":
    args = parse_args()
    main(args)
