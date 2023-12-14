import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Load your dataset
# Assuming df is your DataFrame with the following columns: 'gold_standard', 'model1_prediction', 'model2_prediction'
df = pd.read_csv('/local/scratch/shared-directories/ssanet/SCRIPTS/Test_set_predictions/test_with_ensemble_predictions_hard_soft.csv')
df_dl = pd.read_csv('/local/scratch/shared-directories/ssanet/SCRIPTS/Test_set_predictions/test_op_dl_predictions.csv')
# Calculating Cohen's Kappa Score
# 1. Gold standard and model 1
kappa_gold_model1 = cohen_kappa_score(df['asses'], df['soft_voting_prediction'])

# 2. Gold standard and model 2
kappa_gold_model2 = cohen_kappa_score(df['asses'], df_dl['dl_model_predictions'])

kappa_model1_model2 = cohen_kappa_score(df['soft_voting_prediction'], df_dl['dl_model_predictions'])

# 3. Model 1 and model 2
# kappa_model1_model2 = cohen_kappa_score(df['model1_prediction'], df['soft_voting_prediction'])

# Print the results
print(f"Cohen's Kappa (Gold Standard vs Machine learning): {kappa_gold_model1}")
print(f"Cohen's Kappa (Gold Standard vs Deep learning): {kappa_gold_model2}")
print(f"Cohen's Kappa (ML vs DL): {kappa_model1_model2}")

# 4. Among Gold Standard, Model 1 and Model 2 (Not directly supported by Cohen's Kappa)

# 5. Cohen's Kappa for each class label (Gold Standard vs Model 1, Gold Standard vs Model 2)
# class_labels = df_dl['dl_GT'].unique()

# # Initialize dictionaries to store kappa scores for each model
# kappa_scores_model1 = {}

# for label in class_labels:
#     # Selecting rows where the gold standard is the current label
#     df_label = df_dl[df_dl['dl_GT'] == label]

#     # Calculating Cohen's Kappa for the current label between Gold Standard and Model 1
#     kappa_model1 = cohen_kappa_score(df_label['dl_GT'], df_label['dl_model_predictions'])
#     kappa_scores_model1[label] = kappa_model1

# # Print the results
# print("Cohen's Kappa Scores for each class (Gold Standard vs Model 2 (DL)):")
# for label, kappa in kappa_scores_model1.items():
#     print(f"Class {label}: {kappa}")