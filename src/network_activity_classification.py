from data_processing import process_data_for_network_activity_classification
from utils import get_config, parse_args, save_model
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score


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
    X = data_processed.drop(columns=["target"]).values
    y = data_processed["target"].values
    model = GradientBoostingClassifier(
        n_estimators=10,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_proba)
        auc_scores.append(auc)

    model.fit(X, y)  # Fit on full data after validation
    return model, auc_scores


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
    model, auc_scores = fit_evaluate_model(df_processed)
    save_model(model, "./models/network_activity_classifier.pkl")


if __name__ == "__main__":
    args = parse_args()
    main(args)
