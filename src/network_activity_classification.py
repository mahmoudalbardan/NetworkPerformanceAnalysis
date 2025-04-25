from data_processing import process_data_for_network_activity_classification
from utils import get_config, parse_args,save_model
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
import numpy as np



def fit_evaluate_model(data_processed):
    X = data_processed.drop(columns=["target"]).values
    y = data_processed["target"].values
    model = GradientBoostingClassifier(n_estimators=10,
                                       learning_rate=0.1,
                                       max_depth=5,
                                       random_state=42)
    skf = StratifiedKFold(n_splits=3,
                          shuffle=True,
                          random_state=42)

    auc_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_proba)
        auc_scores.append(auc)

    model.fit(X,y)
    return model, auc_scores





def main(args):
    config = get_config(args.configuration)
    df_processed = process_data_for_network_activity_classification(config)
    print(df_processed.columns, df_processed.shape)
    model, auc_scores = fit_evaluate_model(df_processed)
    print(auc_scores)
    save_model(model, "./src/network_activity_classifier.pkl")


if __name__ == "__main__":
    args = parse_args()
    main(args)