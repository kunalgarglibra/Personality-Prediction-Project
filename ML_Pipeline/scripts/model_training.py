import pandas as pd
import mlflow
import mlflow.xgboost
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
#Metrics
from sklearn.metrics import accuracy_score

#Models
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Load data
df = pd.read_csv("data/processed_data.csv")

# Converting labels to Numeric Data:
# Converting MBTI personality (or target or Y feature) into numerical form using Label Encoding
# encoding personality type
# from sklearn import preprocessing
enc = preprocessing.LabelEncoder()
df['type of encoding'] = enc.fit_transform(df['type'])

Y = df['type of encoding']

# Vectorisation
# from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
# Converting posts (or training or X feature) into numerical form by count vectorization
X =  vect.fit_transform(df["posts"])

x_train, x_test, y_train, y_test = train_test_split(X, df["type of encoding"], test_size=0.2, stratify=Y, random_state=42)
# print ((x_train.shape),(y_train.shape),(x_test.shape),(y_test.shape))

accuracies = {}

#Random Forest
random_forest = RandomForestClassifier(n_estimators=100, random_state = 1)
random_forest.fit(x_train, y_train)

# make predictions for test data
Y_pred = random_forest.predict(x_test)
predictions = [round(value) for value in Y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
accuracies['Random Forest'] = accuracy* 100.0 
print("Random Forest Moodel Accuracy: %.2f%%" % (accuracy * 100.0))
# print(confusion_matrix(y_test, Y_pred))

# Log model with MLflow
mlflow.sklearn.log_model(random_forest, "random_forest_model")
mlflow.log_params({"n_estimators": 100, "random_state": 1})
mlflow.log_metric("accuracyRF", accuracy)
print("RF Model logged successfully!")



#XG boost Classifier
xgb = XGBClassifier()
xgb.fit(x_train,y_train)

Y_pred = xgb.predict(x_test)
predictions = [round(value) for value in Y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
accuracies['XG Boost'] = accuracy* 100.0
print("XG Boost Model Accuracy: %.2f%%" % (accuracy * 100.0))

# Log model with MLflow
mlflow.xgboost.log_model(xgb, "xgboost_model")
mlflow.log_metric("accuracyXGB", accuracy)
print("XGBoost Model logged successfully!")

# from sklearn.svm import SVC
svm = SVC(random_state = 1)
svm.fit(x_train, y_train)

Y_pred = svm.predict(x_test)

predictions = [round(value) for value in Y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
accuracies['SVM'] = accuracy* 100.0
print("SVM Model Accuracy: %.2f%%" % (accuracy * 100.0))
# confusion_matrix(y_test, predictions)

# Log model with MLflow
mlflow.sklearn.log_model(svm, "SVM_model")
mlflow.log_metric("accuracySVM", accuracy)
print("SVM Model logged successfully!")




























    