import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score, precision_score, recall_score,
                             accuracy_score, f1_score, fbeta_score, log_loss, roc_curve, auc)
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import label_binarize
from itertools import cycle
import joblib  # If you have sklearn version < 0.23
# from joblib import dump, load  # Use this if you have sklearn version >= 0.23


# Function to plot learning curves
def plot_learning_curve(estimator, title, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(20, 5))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    return plt

# Function to plot ROC curves for each class
def plot_roc_curve(estimator, X, y, n_classes, title):
    # Compute ROC curve and ROC area for each class
    y_score = estimator.predict_proba(X)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Average it and compute AUC
    mean_tpr /= n_classes

    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'cyan', 'magenta', 'yellow', 'black', 'pink', 'lightgreen', 'lightblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(f'/local/scratch/shared-directories/ssanet/SCRIPTS/Trained ML Models/roc_curve_{title}.png')
    plt.close()

# DataFrame to store results
results_df = pd.DataFrame(columns=['Model name', 'Accuracy', 'Balanced Accuracy', 'ROC AUC', 'Precision', 'Recall', 'f1_micro', 'f1_macro', 'f2_micro', 'f2_macro', 'logloss'])

# Data preparation
train_df_p = pd.concat([pd.read_csv('/local/scratch/shared-directories/ssanet/SCRIPTS/Preprocessed Datasets/train_preprocessed.csv'),
                        pd.read_csv('/local/scratch/shared-directories/ssanet/SCRIPTS/Preprocessed Datasets/val_preprocessed.csv')], ignore_index=True)
test_df_p = pd.read_csv('/local/scratch/shared-directories/ssanet/SCRIPTS/Preprocessed Datasets/test_preprocessed.csv')
X_train = train_df_p.drop('asses', axis=1)
y_train = train_df_p['asses']
X_test = test_df_p.drop('asses', axis=1)
y_test = test_df_p['asses']

# Preprocessing
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test[X_train.columns])
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Binarize the output for multi-class ROC AUC calculation
y_test_binarized = label_binarize(y_test_encoded, classes=np.unique(y_train_encoded))
n_classes = y_test_binarized.shape[1]

# Function to calculate and store model metrics
def calculate_metrics(model, X_train, y_train, X_test, y_test, model_name, results_df):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
    precision = precision_score(y_test_encoded, pred, average='macro')
    recall = recall_score(y_test_encoded, pred, average='macro')
    balanced_accuracy = balanced_accuracy_score(y_test_encoded, pred)
    results_df.loc[len(results_df.index)] = [model_name, 
                                             accuracy_score(y_test_encoded, pred),
                                             balanced_accuracy,
                                             roc_auc,
                                             precision,
                                             recall,
                                             f1_score(y_test_encoded, pred, average='micro'),
                                             f1_score(y_test_encoded, pred, average='macro'),
                                             fbeta_score(y_test_encoded, pred, beta=2, average='micro'),
                                             fbeta_score(y_test_encoded, pred, beta=2, average='macro'),
                                             log_loss(y_test_encoded, model.predict_proba(X_test))]
    plot_learning_curve(model, f"{model_name} Learning Curve", X_train, y_train, cv=5).savefig(f'/local/scratch/shared-directories/ssanet/SCRIPTS/Trained ML Models/learning_curve_{model_name}.png')
    plot_roc_curve(model, X_test, y_test_binarized, n_classes, f"{model_name} ROC Curve")
    joblib.dump(model, f'/local/scratch/shared-directories/ssanet/SCRIPTS/Trained ML Models/{model_name}_model.pkl')
    print(f"{model_name} model saved!")

    print(f"{model_name} done!")

# Model Evaluations
calculate_metrics(MultinomialNB(alpha=0.01), X_train_scaled, y_train_encoded, X_test_scaled, y_test_binarized, 'Naive Bayes', results_df)
calculate_metrics(DecisionTreeClassifier(criterion='gini', max_depth=10), X_train_scaled, y_train_encoded, X_test_scaled, y_test_binarized, 'Decision Tree', results_df)
calculate_metrics(LogisticRegression(C=10, penalty='l2', max_iter=3000), X_train_scaled, y_train_encoded, X_test_scaled, y_test_binarized, 'Logistic Regression', results_df)
calculate_metrics(RandomForestClassifier(max_depth=None, n_estimators=50), X_train_scaled, y_train_encoded, X_test_scaled, y_test_binarized, 'Random Forest', results_df)
calculate_metrics(KNeighborsClassifier(n_neighbors=7), X_train_scaled, y_train_encoded, X_test_scaled, y_test_encoded, 'KNN', results_df)  # KNN doesn't support predict_proba for ROC-AUC
calculate_metrics(SVC(probability=True, C=1, kernel='rbf'), X_train_scaled, y_train_encoded, X_test_scaled, y_test_binarized, 'SVM', results_df)
calculate_metrics(GradientBoostingClassifier(n_estimators=100, learning_rate=0.1), X_train_scaled, y_train_encoded, X_test_scaled, y_test_binarized, 'Gradient Boosting', results_df)

# Saving the results
results_df.to_csv('/local/scratch/shared-directories/ssanet/SCRIPTS/Trained ML Models/static_model_results.csv', index=False)
print("All models processed and results saved.")
