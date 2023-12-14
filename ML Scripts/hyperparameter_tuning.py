import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, fbeta_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier


################################ Using the preprocessed dataset ###############################################
df_train = pd.read_csv('/local/scratch/shared-directories/ssanet/SCRIPTS/Preprocessed Datasets/train_preprocessed.csv')
df_val= pd.read_csv('/local/scratch/shared-directories/ssanet/SCRIPTS/Preprocessed Datasets/val_preprocessed.csv')

print(df_train.shape, flush=True)
df_train,_ = train_test_split(df_train,train_size=0.4,stratify=df_train['asses'])
print(df_train.shape,flush=True)

X_train = df_train.drop('asses',axis=1)
y_train = df_train['asses']-1

X_test = df_val.drop('asses',axis=1)
y_test = df_val['asses']-1
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test[X_train.columns])
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)


############### Hyperparameters tuning ####################
models = {
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier(),
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=3000),
    'SVM': SVC()
}

param_grids = {
    'Naive Bayes': {'alpha': [0.01, 1, 100]},

    'Logistic Regression': {'C': [0.1, 1, 10],
                            'penalty': ['l2'],
                            'solver': ['liblinear']},

    'SVM': {'C': [0.1, 1, 10]},

    'KNN': {'n_neighbors': [5, 7]},

    'Decision Tree': {'max_depth': [None, 10]},
    
    'Random Forest': {'n_estimators': [50, 100],
                      'max_depth': [None, 10, 30]},
    
    'GradientBoosting': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
}


best_models = {}


for classifier_name, classifier in models.items():
    grid_search = GridSearchCV(classifier, param_grids[classifier_name], cv=5)
    grid_search.fit(X_train, y_train)
    best_models[classifier_name] = grid_search.best_estimator_
    models[classifier_name].set_params(**grid_search.best_params_)
    predictions = grid_search.predict(X_test)
    print(f"Model: {classifier_name}",flush=True)
    print(f"Best Parameters: {grid_search.best_params_}", flush=True)
    print(f"Accuracy: {accuracy_score(y_test, predictions)}",flush=True)
    print(f"Micro F1: {f1_score(y_test, predictions, average='micro')}", flush=True)
    print(f"Macro F1: {f1_score(y_test, predictions, average='macro')}", flush=True)
    print(f"Micro F2: {fbeta_score(y_test, predictions, beta=2, average='micro')}", flush=True)
    print(f"Macro F2: {fbeta_score(y_test, predictions, beta=2, average='macro')}", flush=True)
    print("--------------------------------")
print("\n\n Best models parameters")

print(best_models)