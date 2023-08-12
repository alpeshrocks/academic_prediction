import sys
import joblib
import os
import pandas as pd
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QComboBox, \
    QHBoxLayout, QMessageBox


class DecisionTreePredictor(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Decision Tree Predictor")
        self.setGeometry(100, 100, 600, 400)

        self.init_ui()

        model_path = os.path.join(os.path.dirname(sys.argv[0]), 'decision_tree_model.joblib')
        self.model = joblib.load(model_path)

    def init_ui(self):
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout()

        self.create_input_fields(layout)
        self.create_predict_button(layout)
        self.create_result_label(layout)

        self.central_widget.setLayout(layout)

    def create_input_fields(self, layout):
        self.input_widgets = {}

        # Modify the fields dictionary in the create_input_fields method
        fields = [
            ("Gender", ["M", "F"]),
            ("Region", [
                "East Anglian Region", "East Midlands Region", "Ireland", "London Region",
                "North Region", "North Western Region", "Scotland", "South East Region",
                "South Region", "South West Region", "Wales", "West Midlands Region",
                "Yorkshire Region"
            ]),
            ("Highest Education", [
                "HE Qualification", "A Level or Equivalent", "Lower Than A Level",
                "No Formal quals", "Post Graduate Qualification"
            ]),
            ("IMD Band", [
                "0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%",
                "70-80%", "80-90%", "90-100%"
            ]),
            ("Age Band", ["0-35", "35-55", "55<="]),
            ("Number of Previous Attempts", []),
            ("Studied Credits", []),
            ("Disability", ["N", "Y"]),
            ("Number of Clicks", []),
            ("Score", [])
        ]

        for field, options in fields:
            widget_layout = QHBoxLayout()

            label = QLabel(field + ":", self)
            widget_layout.addWidget(label)

            if options:
                combo_box = QComboBox(self)
                combo_box.addItems(options)
                widget_layout.addWidget(combo_box)
                self.input_widgets[field] = combo_box
            else:
                line_edit = QLineEdit(self)
                widget_layout.addWidget(line_edit)
                self.input_widgets[field] = line_edit

            layout.addLayout(widget_layout)

    def create_predict_button(self, layout):
        self.predict_button = QPushButton("Predict Result", self)
        self.predict_button.clicked.connect(self.predict_result)
        layout.addWidget(self.predict_button)

    def create_result_label(self, layout):
        self.result_label = QLabel("", self)
        layout.addWidget(self.result_label)

    def preprocess_input(self, test_data):
        test_data = pd.get_dummies(test_data, columns=['gender', 'region', 'highest_education', 'imd_band', 'age_band',
                                                       'disability'])
        X = ['num_of_prev_attempts', 'studied_credits', 'No_of_Clicks', 'Score',
             'gender_F', 'gender_M', 'region_East Anglian Region',
             'region_East Midlands Region', 'region_Ireland', 'region_London Region',
             'region_North Region', 'region_North Western Region', 'region_Scotland',
             'region_South East Region', 'region_South Region',
             'region_South West Region', 'region_Wales',
             'region_West Midlands Region', 'region_Yorkshire Region',
             'highest_education_A Level or Equivalent',
             'highest_education_HE Qualification',
             'highest_education_Lower Than A Level',
             'highest_education_No Formal quals',
             'highest_education_Post Graduate Qualification', 'imd_band_0-10%',
             'imd_band_20-30%', 'imd_band_240-70%', 'imd_band_30-40%',
             'imd_band_40-50%', 'imd_band_50-240%', 'imd_band_50-60%',
             'imd_band_60-70%', 'imd_band_70-80%', 'imd_band_80-90%',
             'imd_band_90-100%', 'age_band_0-35', 'age_band_35-55', 'age_band_55<=',
             'disability_N', 'disability_Y']
        # Ensure the columns match between training and test data
        missing_cols = set(X) - set(test_data.columns)
        for col in missing_cols:
            test_data[col] = 0
        print(test_data.head())
        # Reorder columns to match the order during training
        test_data = test_data[X]

        return test_data

    def predict_result(self):
        try:
            input_data = {
                "gender": self.input_widgets["Gender"].currentText(),
                "region": self.input_widgets["Region"].currentText(),
                "highest_education": self.input_widgets["Highest Education"].currentText(),
                "imd_band": self.input_widgets["IMD Band"].currentText(),
                "age_band": self.input_widgets["Age Band"].currentText(),
                "num_of_prev_attempts": self.input_widgets["Number of Previous Attempts"].text(),
                "studied_credits": self.input_widgets["Studied Credits"].text(),
                "disability": self.input_widgets["Disability"].currentText(),
                "No_of_Clicks": self.input_widgets["Number of Clicks"].text(),
                "Score": self.input_widgets["Score"].text()
            }

            # Check for empty fields
            empty_fields = [field for field, value in input_data.items() if not value]
            if empty_fields:
                QMessageBox.critical(self, "Error", f"The following fields are empty: {', '.join(empty_fields)}")
                return

            user_data = self.preprocess_input(pd.DataFrame([input_data], index=[0]))
            user_prediction = self.model.predict(user_data)

            predicted_result = user_prediction[0]
            QMessageBox.information(self, "Prediction Result", f"The predicted result is: {predicted_result}")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

#deciontree, NB, Randomforest, distinction
def main():
    app = QApplication(sys.argv)
    window = DecisionTreePredictor()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
