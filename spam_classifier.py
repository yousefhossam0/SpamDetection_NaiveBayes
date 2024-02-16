import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTextEdit, QPushButton, QMessageBox, QLabel, QGridLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal

class ModelTrainingThread(QThread):
    finished = pyqtSignal(object, str)

    def run(self):
        dataset_path = 'spam_ham_dataset.csv'  # Replace with your dataset path
        df = pd.read_csv(dataset_path)
        X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label_num'], test_size=0.2, random_state=42)

        model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred) #in the testing phase is crucial for understanding the model's performance

        eval_text = f"Accuracy: {accuracy:.2f}\nClassification Report:\n{report}\nConfusion Matrix:\n{matrix}"
        self.finished.emit(model, eval_text)

class SpamClassifier(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = None

    def initUI(self):
        self.setWindowTitle('Spam Email Classifier')
        self.setGeometry(100, 100, 600, 400)

        layout = QGridLayout()

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        central_widget.setLayout(layout)

        self.status_label = QLabel('Welcome to the Spam Email Classifier.\nClick "Train Model" to begin.')
        layout.addWidget(self.status_label, 0, 0, 1, 2)

        self.train_btn = QPushButton('Train Model', self)
        self.train_btn.clicked.connect(self.train_model)
        layout.addWidget(self.train_btn, 1, 0)

        self.textEdit = QTextEdit()
        self.textEdit.setPlaceholderText("Enter email content here...")
        layout.addWidget(self.textEdit, 2, 0, 1, 2)

        self.classify_btn = QPushButton('Classify Email', self)
        self.classify_btn.clicked.connect(self.on_classify)
        layout.addWidget(self.classify_btn, 3, 0)
        self.classify_btn.setEnabled(False)

        self.results_label = QLabel()
        layout.addWidget(self.results_label, 4, 0, 1, 2)

    def train_model(self):
        self.train_btn.setEnabled(False)
        self.status_label.setText("Training the model, please wait...")
        self.thread = ModelTrainingThread()
        self.thread.finished.connect(self.on_model_trained)
        self.thread.start()

    def on_model_trained(self, model, eval_text):
        self.model = model
        self.train_btn.setEnabled(True)
        self.classify_btn.setEnabled(True)
        self.status_label.setText("Model Training Completed")
        self.results_label.setText(eval_text)

    def classify_message(self, message):
        prediction = self.model.predict([message])
        return "safe" if prediction[0] == 1 else "spam"

    def on_classify(self):
        text = self.textEdit.toPlainText()
        if not text.strip():
            QMessageBox.warning(self, 'Warning', 'Please enter some email content to classify.')
            return
        result = self.classify_message(text)
        QMessageBox.information(self, 'Classification Result', f'This email is classified as: {result}')

def main():
    app = QApplication(sys.argv)
    ex = SpamClassifier()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
