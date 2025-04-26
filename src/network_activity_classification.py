from data_processing import process_data_for_network_activity_classification
from utils import get_config, parse_args, save_model
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

def fit_evaluate_model(data_processed):
    """
    Train and evaluate a Gradient Boosting model using Stratified K-Fold cross-validation.

    Parameters
    ----------
    data_processed : pd.DataFrame
        Preprocessed dataset containing features and a binary 'target' column.

    Returns
    -------
    GradientBoostingClassifier
        Trained model on the full dataset.
    list of float
        List of AUC scores for each fold in cross-validation.
    """
    df_X = data_processed.drop(columns=["target"])
    df_y = data_processed["target"]
    X,y  = df_X.values, df_y.values
    model = GradientBoostingClassifier(
        n_estimators=10,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    auc_score_all_folds = []
    f1_score_all_folds = []
    precision_score_all_folds = []
    accuracy_score_all_folds = []
    recall_score_all_folds = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)[:, 1]

        auc_scores = []
        f1_scores = []
        precision_scores = []
        recall_scores = []
        accuracy_scores = []
        for proba_threshold in np.arange(0,1,0.01):
            y_pred = (y_proba>proba_threshold).astype(int)
            f1 = f1_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            accuracy = accuracy_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_proba)


            precision_scores.append(precision)
            accuracy_scores.append(accuracy)
            recall_scores.append(recall)
            auc_scores.append(auc)
            f1_scores.append(f1)

        auc_score_all_folds.append(auc_scores)
        f1_score_all_folds.append(f1_scores)
        recall_score_all_folds.append(recall_scores)
        accuracy_score_all_folds.append(accuracy_scores)
        precision_score_all_folds.append(precision_scores)

    average_auc = np.mean(np.array(auc_score_all_folds), axis=0)
    average_f1 = np.mean(np.array(f1_score_all_folds), axis=0)
    average_precision = np.mean(np.array(precision_score_all_folds), axis=0)
    average_accuracy = np.mean(np.array(accuracy_score_all_folds), axis=0)
    average_recall = np.mean(np.array(recall_score_all_folds), axis=0)
    model.fit(X, y)

    fig, ax = plt.subplots(nrows = 1, ncols = 1,figsize=(15,6))
    ax.plot(np.arange(0,1,0.01), average_auc, label="AUC")
    ax.plot(np.arange(0, 1, 0.01), average_f1, label="F1score")
    ax.plot(np.arange(0, 1, 0.01), average_precision, label="Precision")
    ax.plot(np.arange(0, 1, 0.01), average_accuracy,label="Accuracy")
    ax.plot(np.arange(0, 1, 0.01), average_recall, label="Recall")

    best_threshold = np.arange(0, 1, 0.01)[np.argmax(average_f1)]
    ax.axvline(x=best_threshold, ymin=0, ymax=1, color="black",
               label=f"Best probability threshold for f1 score:{best_threshold:.2f}", linestyle="--")
    ax.text(best_threshold, -0.03, f"{best_threshold:.2f}", color='black', fontsize=9)
    ax.set_title("Metrics")
    ax.set_xlabel(" Prediction probability threshold")
    ax.set_ylabel(" Value")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show()
    return model


def main(args):
    """
    Main function to load config, preprocess data, train and evaluate model, and save the result.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments with at least a 'configuration' attribute.

    Returns
    -------
    None
    """
    config = get_config(args.configuration)
    df_processed = process_data_for_network_activity_classification(config)
    model = fit_evaluate_model(df_processed)
    save_model(model, "./models/network_activity_classifier.pkl")



if __name__ == "__main__":
    args = parse_args()
    main(args)
