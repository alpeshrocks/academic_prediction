import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
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
dt_scores = cross_val_score(dt_model, X_train, y_train, cv=5)  # 5-fold cross-validation
print("Decision Tree Cross-Validation Accuracy: {:.2f}".format(dt_scores.mean()))
dt_model.fit(X_train, y_train)
dt_model_filename = 'decision_tree_kmodel.joblib'
joblib.dump(dt_model, dt_model_filename)

# Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_scores = cross_val_score(rf_model, X_train, y_train, cv=5)  # 5-fold cross-validation
print("Random Forest Cross-Validation Accuracy: {:.2f}".format(rf_scores.mean()))
rf_model.fit(X_train, y_train)
rf_model_filename = 'random_forest_kmodel.joblib'
joblib.dump(rf_model, rf_model_filename)

# Naive Bayes Classifier
nb_model = GaussianNB()
nb_scores = cross_val_score(nb_model, X_train, y_train, cv=5)  # 5-fold cross-validation
print("Naive Bayes Cross-Validation Accuracy: {:.2f}".format(nb_scores.mean()))
nb_model.fit(X_train, y_train)
nb_model_filename = 'naive_bayes_kmodel.joblib'
joblib.dump(nb_model, nb_model_filename)
