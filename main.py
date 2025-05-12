import sys
import torch
import torch.nn as nn
import pandas as pd
from PySide6.QtCore import Qt, QPoint, QRect
from PySide6.QtGui import QPainter, QPen, QMouseEvent, QPaintEvent
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QMainWindow, QLabel

from datasets import load_dataset  # Se non lo usi puoi anche rimuoverlo

# --- Modello MNIST in PyTorch ---
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classi per MNIST

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Appiattire l'immagine 28x28
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Funzione per convertire il file parquet in pth
def convert_parquet_to_pth(parquet_file, pth_file):
    df = pd.read_parquet(parquet_file)
    values = df.select_dtypes(include=["number"]).values.flatten()
    weights = torch.tensor(values, dtype=torch.float32)

    state_dict = {
        'fc1.weight': weights[:128 * 28 * 28].reshape(128, 28 * 28),
        'fc1.bias': weights[128 * 28 * 28:128 * 28 * 28 + 128],
        'fc2.weight': weights[128 * 28 * 28 + 128:128 * 28 * 28 + 128 + 10 * 128].reshape(10, 128),
        'fc2.bias': weights[128 * 28 * 28 + 128 + 10 * 128:]
    }

    torch.save(state_dict, pth_file)
    print(f"File {pth_file} salvato con successo!")

# --- Finestra di disegno ---
class DrawingGrid(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(28 * 15, 28 * 15)
        self.drawing = False
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(p)

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setPen(Qt.NoPen)
        painter.setBrush(Qt.black)
        for r in range(28):
            for c in range(28):
                if riceve[r][c] == 1:
                    rect = QRect(c * 15, r * 15, 15, 15)
                    painter.fillRect(rect, Qt.black)

        painter.setPen(QPen(Qt.lightGray, 1))
        for i in range(29):
            x = i * 15
            y = i * 15
            painter.drawLine(x, 0, x, 28 * 15)
            painter.drawLine(0, y, 28 * 15, y)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self._UpdateCell(event.position().toPoint())

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drawing and (event.buttons() & Qt.LeftButton):
            self._UpdateCell(event.position().toPoint())

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def _UpdateCell(self, pos: QPoint):
        c = pos.x() // 15
        r = pos.y() // 15
        if 0 <= r < 28 and 0 <= c < 28:
            if riceve[r][c] == 0:
                riceve[r][c] = 1
                self.update()

    def ClearGrid(self):
        global riceve
        changed = False
        for r in range(28):
            for c in range(28):
                if riceve[r][c] != 0:
                    riceve[r][c] = 0
                    changed = True
        if changed:
            self.update()

# --- Finestra principale ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning")
        self.setFixedSize(400, 400)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        self.drawing_grid = DrawingGrid(central_widget)
        self.drawing_grid.move(10, 10)

        self.recognize_button = QPushButton("Riconosci", central_widget)
        self.recognize_button.setGeometry(300, 30, 90, 40)
        self.recognize_button.clicked.connect(self._HandleRecognize)

        self.clear_button = QPushButton("Cancella", central_widget)
        self.clear_button.setGeometry(300, 80, 90, 40)
        self.clear_button.clicked.connect(self._HandleClear)

        self.save_button = QPushButton("Salva Pesi", central_widget)
        self.save_button.setGeometry(300, 130, 90, 40)
        self.save_button.clicked.connect(self._saveWeightsButton)

        self.load_button = QPushButton("Carica Pesi", central_widget)
        self.load_button.setGeometry(300, 180, 90, 40)
        self.load_button.clicked.connect(self._loadWeightsButton)

        self.result_label = QLabel("Risultato: ", central_widget)
        self.result_label.setGeometry(300, 240, 90, 40)

    def _HandleRecognize(self):
        global mnist_model
        if mnist_model is None:
            print("Modello non caricato!")
            self.result_label.setText("Risultato: N/A")
            return

        input_tensor = torch.tensor(riceve, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = mnist_model(input_tensor)
            _, predicted = torch.max(output, 1)
            prediction = predicted.item()
            print(f"Predizione: {prediction}")
            self.result_label.setText(f"Risultato: {prediction}")

    def _HandleClear(self):
        self.drawing_grid.ClearGrid()
        self.result_label.setText("Risultato: ")

    def _saveWeightsButton(self):
        convert_parquet_to_pth("train-00000-of-00001.parquet", "mnist_model.pth")

    def _loadWeightsButton(self):
        global mnist_model
        mnist_model = MNISTModel()
        state_dict = torch.load("mnist_model.pth", map_location=torch.device('cpu'))
        mnist_model.load_state_dict(state_dict)
        mnist_model.eval()
        print("Modello caricato correttamente.")

# --- Dati Globali ---
riceve = [[0 for _ in range(28)] for _ in range(28)]
mnist_model = None  # Inizializza il modello come variabile globale

# --- Avvio della finestra ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
