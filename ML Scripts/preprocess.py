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


# Read datasets
train_df = pd.read_csv('/local/scratch/shared-directories/ssanet/SCRIPTS/train.csv')
test_df = pd.read_csv('/local/scratch/shared-directories/ssanet/SCRIPTS/test.csv')
validation_df = pd.read_csv('/local/scratch/shared-directories/ssanet/SCRIPTS/val.csv')
# combined_df = pd.concat([train_df, test_df, validation_df], ignore_index=True)


# # Use following code to reduce dataset to smaller subsample
# def stratified_sample(df, target, n_samples_per_class):
#     return df.groupby(target, group_keys=False).apply(lambda x: x.sample(min(len(x), n_samples_per_class)))

# train_df = stratified_sample(train_df, 'asses', n_samples_per_class=10)
# test_df = stratified_sample(test_df, 'asses', n_samples_per_class=5)
# validation_df = stratified_sample(validation_df, 'asses', n_samples_per_class=3)


def preprocess_dataframe(df,data_name):

    # 'GENDER_DESC' can be dropped since we have a similar column called 'PatientSex'
    cols_to_drop = ['desc', 'numfind', 'GENDER_DESC','study_date_anon_x','procdate_anon', 'cohort_num_x', 'first_3_zip', 'cohort_num_y', 'study_date_anon_y', 'StudyDescription', 'SeriesDescription', 'FinalImageType','ViewPosition', 'spot_mag', 'ROI_coords', 'SRC_DST', 'match_level', 'Manufacturer', 'ManufacturerModelName', 'ProtocolName', 'SeriesNumber','SeriesTime', 'StudyID', 'WindowCenter', 'WindowWidth', 'has_pix_array','category', 'VOILUTFunction', 'WindowCenterWidthExplanation']
    df.drop(cols_to_drop, axis=1,inplace=True)

    df['bside'].fillna('LR', inplace=True)
    df['tissueden'].fillna(-1, inplace=True)
    df['side'].fillna('LR', inplace=True)
    df['RACE_DESC'].fillna('Not Recorded', inplace=True)
    df['ETHNIC_GROUP_DESC'].fillna('Not Recorded', inplace=True)
    df['MARITAL_STATUS_DESC'].fillna('Unknown', inplace=True)
    df['age_at_study'].fillna(-1, inplace = True)
    df['path_severity'].fillna(-1, inplace=True)

    def convert_to_list(string):
        return string.strip("[]").replace("'", "").split(", ")

    def unique_sex_value(input_list):
        # In case of a mismatch, find the most frequently occurring element
        if len(set(input_list)) > 1:
            return max(set(input_list), key = input_list.count)
        else:
            # If all elements are the same, return the single unique element
            return input_list[0]

    df['ImageLateralityFinal'] = df['ImageLateralityFinal'].apply(convert_to_list)
    df['num_roi'] = df['num_roi'].apply(convert_to_list)
    df['num_roi'] = df['num_roi'].apply(lambda x: [pd.to_numeric(i) for i in x])
    df['num_roi'] = df['num_roi'].apply(lambda x: sum(x))

    df['PatientSex'] = df['PatientSex'].apply(convert_to_list)
    df['PatientSex'] = df['PatientSex'].apply(unique_sex_value)

    df['anon_dicom_path'] = df['anon_dicom_path'].apply(convert_to_list)
    df['anon_dicom_path'] = df['anon_dicom_path'].apply(len)

    # Function to count occurrences of each number and return in a dict
    def count_numbers(lst):
        count_dict = {}
        for num in lst:
            count_dict[num] = count_dict.get(num, 0) + 1
        return count_dict

    # # Applying the function and creating new columns
    # for index, row in df.iterrows():
    #     counts = count_numbers(row['num_roi'])
    #     for key, value in counts.items():
    #         col_name = f'num_roi_{key}'
    #         df.at[index, col_name] = value

    # df.fillna(0, inplace=True)  # Fill NaN values with 0

    for index, row in df.iterrows():
        counts = count_numbers(row['ImageLateralityFinal'])
        for key, value in counts.items():
            col_name = f'ImageLateralityFinal_{key}'
            df.at[index, col_name] = value

    df.fillna(0, inplace=True)  # Fill NaN values with 0


    # Convert categorical columns to one-hot encoding
    columns_to_encode = ['side','bside','RACE_DESC','ETHNIC_GROUP_DESC','MARITAL_STATUS_DESC','path_severity','PatientSex']

    # Function to apply LabelEncoder to a column
    def label_encode(column):
        le = LabelEncoder()
        return le.fit_transform(column)

    # Apply LabelEncoder only to specified columns
    for column in columns_to_encode:
        df[column] = label_encode(df[column])
    # df_final = pd.get_dummies(df, columns=categorical_cols)

    # Remove unnecessary columns now
    df.drop(['empi_anon', 'acc_anon','ImageLateralityFinal'],axis=1,inplace=True)

    # print(df.columns)
    # print(df.sample(2))

    df['asses'].replace({'A':0,'N':1, 'B':2, 'M':3},inplace=True)
    # df.to_csv(f'temp{data_name}.csv',index=False)
    # print(f"{data_name} Target variable distribution:")
    # Count the occurrences of each class
    class_counts = df['asses'].value_counts()

    # Create a pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(f'{data_name} Class Distribution')

    # Save the plot
    plt.savefig(f'/local/scratch/shared-directories/ssanet/SCRIPTS/{data_name}_class_distribution_pie_chart.png')
    return df

train_df_p = preprocess_dataframe(train_df,"Train Dataset")
test_df_p = preprocess_dataframe(test_df,"Test Dataset")
validation_df_p = preprocess_dataframe(validation_df,"Validation Dataset")

train_df_p.to_csv('/local/scratch/shared-directories/ssanet/SCRIPTS/Preprocessed Datasets/train_preprocessed.csv',index=False)
test_df_p.to_csv('/local/scratch/shared-directories/ssanet/SCRIPTS/Preprocessed Datasets/test_preprocessed.csv',index=False)
validation_df_p.to_csv('/local/scratch/shared-directories/ssanet/SCRIPTS/Preprocessed Datasets/val_preprocessed.csv',index=False)


# print("Train set columns:",train_df_p.columns)
# print("Test set columns:",test_df_p.columns)
# print("val set columns:",validation_df_p.columns)

# Assuming 'asses' is the target and the rest are features
X_train = train_df_p.drop('asses', axis=1)
y_train = train_df_p['asses']
X_val = validation_df_p.drop('asses', axis=1)
y_val = validation_df_p['asses']
X_test = test_df_p.drop('asses', axis=1)
y_test = test_df_p['asses']

# Preprocess the data
# Encode categorical variables if any
# This script assumes all features are numeric
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Align the columns of X_val and X_test with X_train
X_val = X_val[X_train.columns]
X_test = X_test[X_train.columns]

# Now, transform X_val and X_test
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Encode target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

# Hyperparameter grids for tuning
param_grids = {
    'Logistic Regression': {'C': [0.1, 1, 10]},
    'KNN': {'n_neighbors': [3, 5, 7]},
    'Decision Tree': {'max_depth': [3, 5, 7]},
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'Gradient Boosting': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
}

# Function to perform grid search
def grid_search_model(model, params, X_train, y_train, X_val, y_val):
    grid_search = GridSearchCV(model, params, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Evaluate on validation set
    y_pred = best_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    clf_report = classification_report(y_val, y_pred)
    conf_matrix = confusion_matrix(y_val, y_pred)

    return best_model, best_score, accuracy, clf_report, conf_matrix, best_params

# Train and tune models
best_models = {}
for name, model in {'Logistic Regression': LogisticRegression(max_iter=10000),
                    'KNN': KNeighborsClassifier(),
                    'Decision Tree': DecisionTreeClassifier(),
                    'SVM': SVC(),
                    'Gradient Boosting': GradientBoostingClassifier()}.items():
    print(f"Tuning {name}...")
    best_model, best_score, val_accuracy, clf_report, conf_matrix,params_best = grid_search_model(
        model, param_grids[name], X_train_scaled, y_train_encoded, X_val_scaled, y_val_encoded
    )
    best_models[name] = best_model
    print(f"best params for {name}: {params_best}")
    print(f"Score baed on best Params for {name}: {best_score}")
    print(f"Validation Accuracy of {name}: {val_accuracy}\n")
    print(f"Validation Classification Report of {name}:\n{clf_report}\n")
    print(f"Validation Confusion Matrix of {name}:\n{conf_matrix}\n")

# Choose the best model based on the validation performance
best_model_name = max(best_models, key=lambda name: accuracy_score(y_val_encoded, best_models[name].predict(X_val_scaled)))
best_model = best_models[best_model_name]

# Evaluate the best model on the test dataset
print(f"Evaluating the best model ({best_model_name}) on test data...")
y_pred_test = best_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test_encoded, y_pred_test)
test_clf_report = classification_report(y_test_encoded, y_pred_test)
test_conf_matrix = confusion_matrix(y_test_encoded, y_pred_test)

print(f"Test Accuracy of {best_model_name}: {test_accuracy}\n")
print(f"Test Classification Report of {best_model_name}:\n{test_clf_report}\n")
print(f"Test Confusion Matrix of {best_model_name}:\n{test_conf_matrix}\n")