import pandas as pd
import numpy as np
from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier
import pickle
import warnings
warnings.filterwarnings("ignore")


diab1 = pd.read_csv('dataset/diabetes_1.csv')
diab2 = pd.read_csv('dataset/diabetes_2.csv')

diab2 = diab2.drop(columns=['PatientID'])
diab2 = diab2.rename(
    columns={"PlasmaGlucose": "Glucose", "DiastolicBloodPressure": "BloodPressure", "TricepsThickness": "SkinThickness",
             "SerumInsulin": "Insulin", "DiabetesPedigree": "DiabetesPedigreeFunction", "Diabetic": "Outcome"})

diabetes = pd.concat([diab1, diab2]).reset_index(drop=True)

diabetes['Glucose'] = diabetes['Glucose'][(np.abs(stats.zscore(diabetes['Glucose'])) < 3)]
diabetes['SkinThickness'] = diabetes['SkinThickness'][(np.abs(stats.zscore(diabetes['SkinThickness'])) < 3)]
diabetes['BMI'] = diabetes['BMI'][(np.abs(stats.zscore(diabetes['BMI'])) < 3)]
diabetes['BloodPressure'] = diabetes['BloodPressure'][(np.abs(stats.zscore(diabetes['BloodPressure'])) < 3)]

features = ['DiabetesPedigreeFunction', 'Insulin', 'Age', 'Pregnancies']
for item in features:
    percentile25 = diabetes[item].quantile(0.25)
    percentile75 = diabetes[item].quantile(0.75)
    iqr = percentile75 - percentile25
    upp = 1.5 * iqr + percentile75
    low = percentile25 - 1.5 * iqr
    diabetes = diabetes[diabetes[item] < upp]
    diabetes = diabetes[diabetes[item] > low]

diabetes[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = diabetes[
    ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)

train, test = train_test_split(diabetes, test_size=0.25, random_state=20, stratify=diabetes['Outcome'])
train_X = train[train.columns[:8]]
test_X = test[test.columns[:8]]
train_Y = train['Outcome']
test_Y = test['Outcome']

res = np.zeros([5, 2])
kfold = KFold(n_splits=5)

# gamma = np.logspace(-3, 3, 7, base=2)
# c = np.logspace(-3, 3, 7, base=2)
# tol = np.logspace(-10, -1, 10)
# sc =  StandardScaler()


# parameters_svm = {
#     'rns__sampling_strategy': [0.6],
#     'smote__sampling_strategy': [0.9],
#     'svm__gamma'   : [0.25],
#     'svm__C'       : [2],
#     'svm__kernel'  : ['rbf', 'linear', 'sigmoid', 'poly'],
#     # 'svm__tol'     : [1e-1, 1e-2, 1e-4, 1e-5, 1e-10],
#     # 'svm__degree'  : [2, 3, 5, 8, 10],
# }

parameters_svm = {
    'smote__sampling_strategy': [0.9],
    # 'pca__n_components': [8],
    'svm__gamma': [0.25],
    'svm__C': [2],
    'svm__kernel': ['rbf'],
    'svm__tol': [0.1],
}

model = Pipeline([
    ('scaler', StandardScaler()),
    ('imputer', KNNImputer()),
    ('smote', SMOTETomek()),
    # ('pca', pca()),
    ('svm', svm.SVC())
])

# grid = RandomizedSearchCV(model, parameters_svm, cv=kfold, n_jobs=-1, verbose=1)
grid = GridSearchCV(model, param_grid=parameters_svm, cv=kfold, scoring='accuracy',
                    return_train_score=True, n_jobs=-1)
grid.fit(train_X, train_Y)
# print('Accuracy: ', grid.best_score_)
# print(grid.best_params_)

# y_test_predict = grid.best_estimator_.predict(test_X)
# print('Classification report for Pipeline: ')
# print(classification_report(test_Y, y_test_predict))

# RocCurveDisplay.from_estimator(grid.best_estimator_, test_X, test_Y)
# res[0][0] = grid.best_estimator_.score(train_X, train_Y)
# res[0][1] = grid.best_estimator_.score(test_X, test_Y)
svm_model = grid.best_estimator_['svm']


# c = np.logspace(-4, 4, 9, base=2)
# tol = np.logspace(-1, -10, 10)
# solver = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
kfold = KFold(n_splits=5)

# parameters_log = {
#     'rns__sampling_strategy': [0.6],
#     'smote__sampling_strategy': [0.9],
#     'penalty' : ['l1','l2', 'elasticnet'],
#     'C'       : [10,100,1000],
#     'solver'  : ['lbfgs’, 'liblinear’, 'newton-cg’, 'newton-cholesky’, 'sag’, 'saga’],
#     'tol'     : [1e-1, 1e-2]
# }

parameters_log = {
    'smote__sampling_strategy': [0.9],
    # 'pca__n_components': [8],
    'log__penalty': ['l1'],
    'log__C': [1 / 2],
    'log__solver': ['saga'],
    'log__tol': [1e-01]
}

model = Pipeline([
    ('scaler', StandardScaler()),
    ('imputer', KNNImputer()),
    ('smote', SMOTETomek()),
    # ('pca', PCA()),
    ('log', LogisticRegression())
])

# grid = RandomizedSearchCV(model, parameters_log, cv=kfold, n_jobs=-1, verbose=3)
grid = GridSearchCV(model, param_grid=parameters_log, cv=kfold, scoring='accuracy',
                    return_train_score=True, n_jobs=-1, verbose=3)
grid.fit(train_X, train_Y)
# print('Accuracy: ', grid.best_score_)
# print(grid.best_params_)

# y_test_predict_log = grid.best_estimator_.predict(test_X)
# print('Classification report for Pipeline: ')
# print(classification_report(test_Y, y_test_predict))

# RocCurveDisplay.from_estimator(grid.best_estimator_, test_X, test_Y)
# RocCurveDisplay.from_estimator(log, test_X, test_Y)
# res[1][0] = grid.best_estimator_.score(train_X, train_Y)
# res[1][1] = grid.best_estimator_.score(test_X, test_Y)
log_model = grid.best_estimator_['log']


kfold = KFold(n_splits=5)

# parameters_dt = {
#     'rns__sampling_strategy': [0.6],
#     'smote__sampling_strategy': [0.9],
#     'dt__n_estimators': 500),
#     'dt__max_depth': list(np.arange(1, 300, 30)),
#     'dt__min_samples_split': list(np.arange(2, 18, 4)),
#     'dt__min_samples_leaf': list(np.arange(1, 21, 4)),
#     'dt__criterion': ['entropy', 'gini', 'log_loss'],
#     'dt__class_weight': ['balanced'],
# }

parameters_dt = {
    'smote__sampling_strategy': [0.9],
    # 'pca__n_components': [8],
    'dt__min_samples_split': [20],
    'dt__min_samples_leaf': [5],
    'dt__criterion': ['entropy'],
    'dt__max_depth': [35],
    'dt__class_weight': ['balanced'],
}

model = Pipeline([
    ('scaler', StandardScaler()),
    ('imputer', KNNImputer()),
    ('smote', SMOTETomek()),
    # ('pca', PCA()),
    ('dt', DecisionTreeClassifier())
])
# grid = RandomizedSearchCV(model, parameters_dt, cv=kfold, n_jobs=-1)
grid = GridSearchCV(model, param_grid=parameters_dt, cv=kfold, scoring='accuracy',
                    return_train_score=True, n_jobs=-1)
grid.fit(train_X, train_Y)
# print('Accuracy: ', grid.best_score_)
# print(grid.best_params_)

# y_test_predict_dt = grid.best_estimator_.predict(test_X)
# print('Classification report for Pipeline: ')
# print(classification_report(test_Y, y_test_predict))


# RocCurveDisplay.from_estimator(grid.best_estimator_, test_X, test_Y)
# RocCurveDisplay.from_estimator(dt, test_X, test_Y)
# res[2][0] = grid.best_estimator_.score(train_X, train_Y)
# res[2][1] = grid.best_estimator_.score(test_X, test_Y)
dt_model = grid.best_estimator_['dt']

kfold = KFold(n_splits=5)


# parameters_rf = {
#     'rns__sampling_strategy': [0.6],
#     'smote__sampling_strategy': [0.9],
#     'rf__n_estimators': 500),
#     'rf__max_depth': list(np.arange(1, 300, 30)),
#     'rf__min_samples_split': list(np.arange(2, 18, 4)),
#     'rf__min_samples_leaf': list(np.arange(1, 21, 4)),
#     'rf__criterion': ['entropy', 'gini', 'log_loss'],
#     'rf__class_weight': ['balanced'],
# }

parameters_rf = {
    'smote__sampling_strategy': [0.9],
    # 'pca__n_components': [8],
    'rf__n_estimators': [520],
    'rf__max_depth': [190],
    'rf__min_samples_split': [10],
    'rf__min_samples_leaf': [2],
    'rf__criterion': ['entropy'],
    'rf__class_weight': ['balanced'],
}

model = Pipeline([
    ('scaler', StandardScaler()),
    ('imputer', KNNImputer()),
    ('smote', SMOTETomek()),
    # ('pca', PCA()),
    ('rf', RandomForestClassifier())
])

# grid = RandomizedSearchCV(model, parameters_rf, cv=kfold, n_jobs=-1)
grid = GridSearchCV(model, param_grid=parameters_rf, cv=kfold, scoring='accuracy',
                    return_train_score=True, n_jobs=-1)
grid.fit(train_X, train_Y)
# print('Best accuracy: ', grid.best_score_)
# print(grid.best_params_)

# y_test_predict_rf = grid.best_estimator_.predict(test_X)
# print('Classification report for Pipeline: ')
# print(classification_report(test_Y, y_test_predict))

# RocCurveDisplay.from_estimator(grid.best_estimator_, test_X, test_Y)
# res[3][0] = grid.best_estimator_.score(train_X, train_Y)
# res[3][1] = grid.best_estimator_.score(test_X, test_Y)
rf_model = grid.best_estimator_['rf']


kfold = KFold(n_splits=5)

# parameters_knn = {
#     'n_neighbors': np.arange(1, 31, 1),
#     'algorithm': ['ball_tree', 'kd_tree', 'brute'],
#     'leaf_size': [3, 5, 30, 60],
#     'metric': ['manhattan', 'cityblock', 'cosine', 'euclidean', 'haversine', 'l1', 'l2', 'nan_euclidean']
# }

parameters_knn = {
    'smote__sampling_strategy': [0.9],
    # 'pca__n_components': [8],
    'knn__n_neighbors': [12],
    'knn__algorithm': ['ball_tree'],
    'knn__leaf_size': [30],
    'knn__metric': ['manhattan']
}

model = Pipeline([
    ('scaler', StandardScaler()),
    ('imputer', KNNImputer()),
    ('smote', SMOTETomek()),
    # ('pca', PCA()),
    ('knn', KNeighborsClassifier())
])

# grid = RandomizedSearchCV(model, parameters_knn, cv=kfold, n_jobs=-1)
grid = GridSearchCV(model, param_grid=parameters_knn, cv=kfold, scoring='accuracy',
                    return_train_score=True, n_jobs=-1)
grid.fit(train_X, train_Y)
# print('Accuracy: ', grid.best_score_)
# print(grid.best_params_)

# y_test_predict_knn = grid.best_estimator_.predict(test_X)
# print('Classification report for Pipeline: ')
# print(classification_report(test_Y, y_test_predict))

# RocCurveDisplay.from_estimator(grid.best_estimator_, test_X, test_Y)
# res[4][0] = grid.best_estimator_.score(train_X, train_Y)
# res[4][1] = grid.best_estimator_.score(test_X, test_Y)
knn_model = grid.best_estimator_['knn']


sc = StandardScaler()
train_X = pd.DataFrame(sc.fit_transform(train_X), columns=train_X.columns)
test_X = pd.DataFrame(sc.transform(test_X), columns=test_X.columns)

pickle.dump(sc, open('sc.pkl', 'wb'))
imputer = KNNImputer(n_neighbors=5, weights="uniform")

train_X = pd.DataFrame(imputer.fit_transform(train_X), columns=train_X.columns).reset_index(drop=True)
test_X = pd.DataFrame(imputer.transform(test_X), columns=test_X.columns).reset_index(drop=True)

models_vote = [('svm', svm_model), ('dt', dt_model), ('rf', rf_model), ('knn', knn_model)]
svm_model.probability = True

voting_ensemble = VotingClassifier(estimators=models_vote, voting='soft')
voting_ensemble.fit(train_X, train_Y)
pickle.dump(voting_ensemble, open('voting_ensemble.pkl', 'wb'))
# Make predictions with the ensemble model
# y_pred = ensemble.predict(test_X)

# print('Classification report for Ensemble: ')
# print(classification_report(test_Y, y_pred))
# RocCurveDisplay.from_estimator(ensemble.predict_proba(test_X), test_X, test_Y)

# print(ensemble.score(train_X, train_Y))
# print(ensemble.score(test_X, test_Y))


models_stack = [('knn', knn_model), ('dt', dt_model), ('rf', rf_model), ('svm', svm_model)]
stack_ensemble = StackingClassifier(models_stack, final_estimator=LogisticRegression(), cv=3)
stack_ensemble.fit(train_X, train_Y)
pickle.dump(stack_ensemble, open('stack_ensemble.pkl', 'wb'))
# y_pred = model.predict(test_X)

# print('Classification report for Ensemble: ')
# print(classification_report(test_Y, y_pred))
#
# print(model.score(train_X, train_Y))
# print(model.score(test_X, test_Y))