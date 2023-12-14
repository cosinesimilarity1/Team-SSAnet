import pandas as pd
import numpy as np
import joblib
from scipy.stats import mode
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, fbeta_score, 
                             precision_score, recall_score, roc_auc_score, classification_report, 
                             confusion_matrix, log_loss)

# Load test data
test_data = pd.read_csv('/local/scratch/shared-directories/ssanet/SCRIPTS/Preprocessed Datasets/test_preprocessed.csv')

# List of model filenames
# model_filenames = [
#     '/local/scratch/shared-directories/ssanet/SCRIPTS/Trained ML Models/Decision Tree_model.pkl', 
#     '/local/scratch/shared-directories/ssanet/SCRIPTS/Trained ML Models/Gradient Boosting_model.pkl', 
#     '/local/scratch/shared-directories/ssanet/SCRIPTS/Trained ML Models/KNN_model.pkl', 
#     '/local/scratch/shared-directories/ssanet/SCRIPTS/Trained ML Models/Logistic Regression_model.pkl', 
#     '/local/scratch/shared-directories/ssanet/SCRIPTS/Trained ML Models/Naive Bayes_model.pkl', 
#     '/local/scratch/shared-directories/ssanet/SCRIPTS/Trained ML Models/Random Forest_model.pkl', 
#     '/local/scratch/shared-directories/ssanet/SCRIPTS/Trained ML Models/SVM_model.pkl'
# ]

# Load all models
# models = [joblib.load(filename) for filename in model_filenames]
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
pd.set_option('display.max_rows', None)  # None means unlimited
pd.set_option('display.max_columns', None)  # None means unlimited
# Load the trained SVM model
# with open('/local/scratch/shared-directories/ssanet/SCRIPTS/Trained ML Models/SVM_model.pkl', 'rb') as file:
    # svm_model = pickle.load(file)
svm_model = joblib.load('/local/scratch/shared-directories/ssanet/SCRIPTS/Trained ML Models/SVM_model.pkl')

# Assuming X_test is the feature set from test_data
X_test = test_data.drop('asses', axis=1) # Assuming 'asses' is the target column
test_data['svm_predictions'] = svm_model.predict(X_test)
test_data.to_csv("/local/scratch/shared-directories/ssanet/SCRIPTS/Test_set_predictions/svm_predictions.csv",index=False)
print("Predictions saved!")
# Collect predictions from each model for hard voting
# hard_predictions = [model.predict(X_test) for model in models]
# hard_voting_results = mode(hard_predictions, axis=0)[0][0]

# # Collect probability predictions for soft voting
# soft_predictions = [model.predict_proba(X_test) for model in models if hasattr(model, 'predict_proba')]
# if soft_predictions:
#     # Average the probabilities and take the argmax as the prediction
#     soft_voting_prob = np.mean(np.array(soft_predictions), axis=0)
#     soft_voting_results = np.argmax(soft_voting_prob, axis=1)
# else:
#     # Fallback to hard voting if any model doesn't support predict_proba
#     soft_voting_results = hard_voting_results  

# # Add voting predictions to the test data
# test_data['hard_voting_prediction'] = hard_voting_results
# test_data['soft_voting_prediction'] = soft_voting_results

# # Save the updated test data to a new CSV file
# test_data.to_csv('/local/scratch/shared-directories/ssanet/SCRIPTS/Test_set_predictions/test_with_ensemble_predictions_hard_soft.csv', index=False)

# # Assuming 'y_true' contains the true labels
# y_true = test_data['asses']
# y_pred = test_data['soft_voting_prediction']

# print(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
# print(f"Classification Report:\n{classification_report(y_true, y_pred)}")

# # Append results to the final results DataFrame
# final_results = pd.read_csv('/local/scratch/shared-directories/ssanet/SCRIPTS/static_model_results.csv')
# final_results.loc[len(final_results.index)] = ['Ensemble model', 
#                                                accuracy_score(y_true, y_pred),
#                                                balanced_accuracy_score(y_true, y_pred),
#                                                roc_auc_score(y_true, soft_voting_prob, multi_class='ovr'),
#                                                precision_score(y_true, y_pred, average='macro'),
#                                                recall_score(y_true, y_pred, average='macro'),
#                                                f1_score(y_true, y_pred, average='micro'),
#                                                f1_score(y_true, y_pred, average='macro'),
#                                                fbeta_score(y_true, y_pred, beta=2, average='micro'),
#                                                fbeta_score(y_true, y_pred, beta=2, average='macro'),
#                                                log_loss(y_true, soft_voting_prob)]

# final_results.to_csv('/local/scratch/shared-directories/ssanet/SCRIPTS/static_model_results_with_ensemble.csv', index=False)