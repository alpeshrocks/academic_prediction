import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv('ACCURACY_100_FINAL.csv')  # Replace 'your_data.csv' with the actual filename
# Drop columns that are not needed for prediction
data.drop(['id_student', 'code_module', 'code_presentation'], axis=1, inplace=True)

# Convert categorical variables to numerical using one-hot encoding
data = pd.get_dummies(data, columns=['gender', 'region', 'highest_education', 'imd_band', 'age_band', 'disability'])

# Split data into features (X) and target (y)
X = data.drop('final_result', axis=1)
y = data['final_result']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
model_filename = 'decision_tree_model.joblib'
joblib.dump(model, model_filename)