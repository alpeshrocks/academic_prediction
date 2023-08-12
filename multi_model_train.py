import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data = pd.read_csv('ACCURACY_100_FINAL.csv')

# Drop columns that are not needed for prediction
data.drop(['id_student', 'code_module', 'code_presentation'], axis=1, inplace=True)

# Convert categorical variables to numerical using one-hot encoding
data = pd.get_dummies(data, columns=['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability'])

# Split data into features (X) and target (y)
X = data.drop('final_result', axis=1)
y = data['final_result']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt:.2f}")
dt_model_filename = 'decision_tree_mmodel.joblib'
joblib.dump(dt_model, dt_model_filename)

# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {accuracy_rf:.2f}")
rf_model_filename = 'random_forest_mmodel.joblib'
joblib.dump(rf_model, rf_model_filename)

# Naive Bayes Classifier
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"Naive Bayes Accuracy: {accuracy_nb:.2f}")
nb_model_filename = 'naive_bayes_mmodel.joblib'
joblib.dump(nb_model, nb_model_filename)
