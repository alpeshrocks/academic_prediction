import sys
import pandas as pd
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget
import joblib
import traceback


class ModelSelectionWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Model Selection")
        self.setGeometry(100, 100, 300, 200)

        self.layout = QVBoxLayout()
        self.label = QLabel("Select a machine learning model:")
        self.layout.addWidget(self.label)

        self.decision_tree_button = QPushButton("Decision Tree")
        self.decision_tree_button.clicked.connect(self.load_csv_and_process)
        self.layout.addWidget(self.decision_tree_button)

        self.random_forest_button = QPushButton("Random Forest")
        self.random_forest_button.clicked.connect(self.load_csv_and_process)
        self.layout.addWidget(self.random_forest_button)

        self.central_widget = QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

    def load_csv_and_process(self):
        model_name = self.sender().text()
        model_filename = f'{model_name.lower().replace(" ", "_")}_mmodel.joblib'

        try:
            model = joblib.load(model_filename)
        except FileNotFoundError:
            print(f"Model {model_name} not found. Please train and save the model first.")
            return
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        file_dialog = QFileDialog()
        csv_path, _ = file_dialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")

        if not csv_path:
            print("No CSV file selected.")
            return

        try:
            data = pd.read_csv(csv_path)
            df = pd.read_csv(csv_path)
            # Perform the same preprocessing steps as before
            data.drop(['id_student', 'code_module', 'code_presentation'], axis=1, inplace=True)
            data = pd.get_dummies(data, columns=['gender', 'region', 'highest_education', 'imd_band', 'age_band',
                                                 'disability'])

            X = data

            # Predict using the selected model
            y_pred = model.predict(X)

            # Add the predicted 'final_result' column to the data
            data['final_result'] = y_pred
            df['final_result'] = y_pred
            modified_csv_path = csv_path.replace('.csv', f'_{model_name.lower().replace(" ", "_")}_result.csv')
            df.to_csv(modified_csv_path, index=False)
            print(f"Modified CSV with predictions saved as: {modified_csv_path}")
        except Exception as e:
            print(f"Error processing CSV: {e}")
            traceback.print_exc()


def main():
    app = QApplication(sys.argv)
    window = ModelSelectionWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
