import os
import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QMainWindow
)
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QMouseEvent, QPaintEvent
from PySide6.QtCore import Qt, QPoint, QRect, QSize
import functools
import numpy as np

import SavingAndLoading as sal
from Model import Model

# --- Costanti ---
GRID_SIZE = 32
CELL_SIZE = 15
MAIN_GRID_PIXEL_SIZE = GRID_SIZE * CELL_SIZE
# Costanti per il layout della finestra
MAIN_GRID_X = 10
MAIN_GRID_Y = 10
BUTTON_AREA_Y = MAIN_GRID_Y + MAIN_GRID_PIXEL_SIZE + 20
LABEL_AREA_Y1 = BUTTON_AREA_Y + 50
LABEL_AREA_Y2 = BUTTON_AREA_Y + 70
LABEL_AREA_Y3 = BUTTON_AREA_Y + 90
RIGHT_PANEL_X = MAIN_GRID_X + MAIN_GRID_PIXEL_SIZE + 20
WEIGHT_GRID_CELL_SIZE = 3
WEIGHT_GRID_PIXEL_SIZE = GRID_SIZE * WEIGHT_GRID_CELL_SIZE
WEIGHT_GRID_SPACING = 15
WINDOW_WIDTH = RIGHT_PANEL_X + 2 * (WEIGHT_GRID_PIXEL_SIZE + WEIGHT_GRID_SPACING + 80)
WINDOW_HEIGHT = MAIN_GRID_Y + MAIN_GRID_PIXEL_SIZE + 140

WEIGHTS_NAME = "weights.npy"
NUMS_NAME = "nums.npy"

# Inizializza il modello
MODEL = Model(input_shape=(GRID_SIZE, GRID_SIZE), neuronsN=100, outputsN=10, learning_rate=0.001)

# --- Dati Globali ---
# 'riceve' contiene i dati della griglia attualmente disegnata
RICEVE = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
# 'griglia' contiene le matrici dei pesi appresi per ogni cifra (0-9)
GRIGLIA = [[[0.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)] for _ in range(10)]
# 'isTaken' conta quante volte ogni cifra è stata cliccata
IS_TAKEN = [0 for _ in range(10)]
# 'sums' memorizza una somma calcolata relativa al riconoscimento
SUMS = [0.0 for _ in range(10)]
# 'percentuali' è la percentuale di riconoscimento
PERCENTUALI = [0.0 for _ in range(10)]


# --- Widget di Disegno per la Griglia Principale ---
class DrawingGrid(QWidget):
    """Widget per disegnare una griglia binaria di dimensione GRID_SIZE x GRID_SIZE."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(MAIN_GRID_PIXEL_SIZE, MAIN_GRID_PIXEL_SIZE)
        self.drawing = False
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(p)

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)

        # Disegna le celle riempite
        painter.setPen(Qt.NoPen)
        painter.setBrush(Qt.black)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if RICEVE[r][c] == 1:
                    rect = QRect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    painter.fillRect(rect, Qt.black)

        # Disegna le linee della griglia
        painter.setPen(QPen(Qt.lightGray, 1))
        for i in range(GRID_SIZE + 1):
            x = i * CELL_SIZE
            y = i * CELL_SIZE
            painter.drawLine(x, 0, x, MAIN_GRID_PIXEL_SIZE)  # Verticale
            painter.drawLine(0, y, MAIN_GRID_PIXEL_SIZE, y)  # Orizzontale

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self._update_cell(event.position().toPoint())

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.drawing and (event.buttons() & Qt.LeftButton):
            self._update_cell(event.position().toPoint())

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def _update_cell(self, pos: QPoint):
        c = int(pos.x()) // CELL_SIZE
        r = int(pos.y()) // CELL_SIZE
        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE:
            if RICEVE[r][c] == 0:  # Aggiorna solo se cambiato
                RICEVE[r][c] = 1
                self.update()

    def clear_grid(self):
        """Pulisce la griglia di disegno."""
        global RICEVE
        changed = False
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if RICEVE[r][c] != 0:
                    RICEVE[r][c] = 0
                    changed = True
        if changed:
            self.update()


# --- Widget per visualizzare una singola Griglia dei Pesi appresa ---
class WeightGridDisplay(QWidget):
    """Widget per visualizzare la matrice dei pesi appresa per una cifra."""
    def __init__(self, index, parent=None):
        super().__init__(parent)
        self.index = index
        self.setFixedSize(WEIGHT_GRID_PIXEL_SIZE, WEIGHT_GRID_PIXEL_SIZE)
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(p)

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setPen(Qt.NoPen)

        matrix = GRIGLIA[self.index]
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                value = max(0.0, min(1.0, matrix[r][c]))
                gray = int(value * 255)
                color = QColor(gray, gray, gray)
                rect = QRect(c * WEIGHT_GRID_CELL_SIZE, r * WEIGHT_GRID_CELL_SIZE,
                             WEIGHT_GRID_CELL_SIZE, WEIGHT_GRID_CELL_SIZE)
                painter.fillRect(rect, color)

    def update_display(self):
        """Aggiorna la visualizzazione della griglia dei pesi."""
        self.update()


# --- Finestra Principale dell'Applicazione ---
class MainWindow(QMainWindow):
    """Finestra principale dell'applicazione di apprendimento delle cifre."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Riconoscimento Cifre")
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)

        # --- Widget Centrale ---
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # --- Griglia di Disegno ---
        self.drawing_grid = DrawingGrid(central_widget)
        self.drawing_grid.move(MAIN_GRID_X, MAIN_GRID_Y)

        # --- Pulsanti ---
        self.recognize_button = QPushButton("Riconosci", central_widget)
        self.recognize_button.setGeometry(RIGHT_PANEL_X, MAIN_GRID_Y + 20, 90, 40)
        self.recognize_button.clicked.connect(self._handle_recognize)

        self.clear_button = QPushButton("Cancella", central_widget)
        self.clear_button.setGeometry(RIGHT_PANEL_X, MAIN_GRID_Y + 80, 90, 40)
        self.clear_button.clicked.connect(self._handle_clear)

        self.save_weights_button = QPushButton("Salva Pesi", central_widget)
        self.save_weights_button.setGeometry(RIGHT_PANEL_X, MAIN_GRID_Y + 140, 90, 40)
        self.save_weights_button.clicked.connect(self._save_weights_button)

        self.load_weights_button = QPushButton("Carica Pesi", central_widget)
        self.load_weights_button.setGeometry(RIGHT_PANEL_X, MAIN_GRID_Y + 200, 90, 40)
        self.load_weights_button.clicked.connect(self._load_weights_button)

        self.delete_weights_button = QPushButton("Cancella Pesi", central_widget)
        self.delete_weights_button.setGeometry(RIGHT_PANEL_X, MAIN_GRID_Y + 260, 90, 40)
        self.delete_weights_button.clicked.connect(self._delete_weights_button)

        self.number_buttons = []
        for i in range(10):
            button = QPushButton(str(i), central_widget)
            button.setGeometry(MAIN_GRID_X + i * 48, BUTTON_AREA_Y, 40, 40)
            button.clicked.connect(functools.partial(self._handle_number_button, i))
            self.number_buttons.append(button)

        # --- Label ---
        self.is_taken_labels = []
        self.sum_labels = []
        self.recognition_percent = []
        for i in range(10):
            # Label isTaken
            label_taken = QLabel("0", central_widget)
            label_taken.setGeometry(MAIN_GRID_X + i * 48 + 10, LABEL_AREA_Y1, 30, 20)
            label_taken.setAlignment(Qt.AlignCenter)
            self.is_taken_labels.append(label_taken)

            # Label Somma
            label_sum = QLabel("0.0", central_widget)
            label_sum.setGeometry(MAIN_GRID_X + i * 48 + 5, LABEL_AREA_Y2, 40, 20)
            label_sum.setAlignment(Qt.AlignCenter)
            self.sum_labels.append(label_sum)

            # Label Percentuale
            label_percent = QLabel("0.00%", central_widget)
            label_percent.setGeometry(MAIN_GRID_X + i * 48 + 5, LABEL_AREA_Y3, 40, 20)
            label_percent.setAlignment(Qt.AlignCenter)
            self.recognition_percent.append(label_percent)

        # --- Visualizzatori delle Griglie dei Pesi ---
        self.weight_grid_displays = []
        for i in range(10):
            display = WeightGridDisplay(i, central_widget)
            col = i // 5
            row = i % 5
            x_pos = 150 + RIGHT_PANEL_X + col * (WEIGHT_GRID_PIXEL_SIZE + WEIGHT_GRID_SPACING)
            y_pos = 20 + MAIN_GRID_Y + row * (WEIGHT_GRID_PIXEL_SIZE + WEIGHT_GRID_SPACING)
            display.move(x_pos, y_pos)
            self.weight_grid_displays.append(display)

        self._update_number_info()

    # --- Gestori dei Pulsanti ---
    def _handle_recognize(self):
        """Gestisce il pulsante 'Riconosci' per predire la cifra disegnata."""
        global PERCENTUALI
        # Calcola le percentuali basate su GRIGLIA
        for i in range(10):
            contatore = 0
            PERCENTUALI[i] = 0
            for j in range(GRID_SIZE):
                for k in range(GRID_SIZE):
                    PERCENTUALI[i] += RICEVE[j][k] * GRIGLIA[i][j][k]
                    contatore += RICEVE[j][k]
            if contatore > 0:
                PERCENTUALI[i] /= contatore
                PERCENTUALI[i] *= 100
            self.recognition_percent[i].setText(f"{PERCENTUALI[i]:.2f}%")

        # Prepara l'input per il modello
        input_array = np.array(RICEVE, dtype=float) / 1.0  # Normalizza (valori 0 o 1)
        input_batch = input_array[np.newaxis, ...]  # Forma: (1, 32, 32)
        predictions = MODEL.predict(input_batch)[0] * 100  # Converti in percentuali
        predicted_class = np.argmax(predictions)
        print(f"Predizioni modello (%): {predictions}")
        print(f"Classe predetta: {predicted_class}")

    def _handle_clear(self):
        """Gestisce il pulsante 'Cancella' per pulire la griglia di disegno."""
        self.drawing_grid.clear_grid()

    def _handle_number_button(self, index):
        """Gestisce i pulsanti numerici per addestrare il modello."""
        global IS_TAKEN, GRIGLIA, SUMS
        IS_TAKEN[index] += 1
        GRIGLIA[index] = self._calculate_new_matrix(RICEVE, GRIGLIA[index], index)
        SUMS[index] = self._calculate_sum(index)

        # Addestra il modello sul disegno corrente
        input_array = np.array(RICEVE, dtype=float) / 1.0  # Normalizza
        input_batch = input_array[np.newaxis, ...]  # Forma: (1, 32, 32)
        labels = np.array([index], dtype=int)  # Etichetta come array
        loss = MODEL.train(input_batch, labels, max_epochs=1, batch_size=1, patience=1)
        print(f"Addestramento per cifra {index}, perdita: {loss[-1]:.4f}")

        # Aggiorna il visualizzatore della griglia dei pesi
        self.weight_grid_displays[index].update_display()

        # Aggiorna le label
        self._update_number_info()

        # Pulisci la griglia di disegno
        self.drawing_grid.clear_grid()

    # --- Funzioni di Supporto ---
    def _calculate_new_matrix(self, m1, m2, index):
        """Calcola una nuova matrice combinando m1 e m2 (media ponderata)."""
        new_matrix = [[0.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        count = IS_TAKEN[index]
        if count <= 0:
            return m2

        prev_weight = (count - 1) / count
        new_weight = 1 / count

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                new_matrix[i][j] = (m2[i][j] * prev_weight) + (m1[i][j] * new_weight)
                new_matrix[i][j] = max(0.0, min(1.0, new_matrix[i][j]))
        return new_matrix

    def _calculate_sum(self, index):
        """Calcola la somma del prodotto scalare tra 'RICEVE' e 'GRIGLIA[index]'."""
        current_sum = 0.0
        weight_matrix = GRIGLIA[index]
        input_matrix = RICEVE

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                current_sum += input_matrix[i][j] * weight_matrix[i][j]
        return current_sum

    def _update_number_info(self):
        """Aggiorna il testo delle label isTaken e sum."""
        for i in range(10):
            self.is_taken_labels[i].setText(str(IS_TAKEN[i]))
            self.sum_labels[i].setText(f"{SUMS[i]:.2f}")

    def _save_weights_button(self):
        """Gestisce il pulsante 'Salva Pesi'."""
        global GRIGLIA, IS_TAKEN
        # Salva GRIGLIA e IS_TAKEN
        sal.save_weights_and_inserts(GRIGLIA, IS_TAKEN, WEIGHTS_NAME, NUMS_NAME)
        # Salva i pesi del modello
        sal.save_new_model(MODEL, folder_path="Model")

    def _load_weights_button(self):
        """Gestisce il pulsante 'Carica Pesi'."""
        global GRIGLIA, IS_TAKEN
        # Carica GRIGLIA e IS_TAKEN
        weights, inserts = sal.load_weights_and_inserts(WEIGHTS_NAME, NUMS_NAME)
        if weights is not None and inserts is not None:
            GRIGLIA = weights
            IS_TAKEN = inserts
            # Aggiorna i visualizzatori
            for i in range(10):
                self.weight_grid_displays[i].update_display()
            self._update_number_info()

        global MODEL
        # Carica i pesi del modello
        loaded_model = sal.load_new_model(MODEL, folder_path="Model")
        if loaded_model is not None:
            MODEL = loaded_model
            print("Modello neurale caricato con successo")

    def _delete_weights_button(self):
        """Gestisce il pulsante 'Cancella Pesi'."""
        sal.delete_weights(WEIGHTS_NAME)
        # Opzionale: cancellare anche la cartella del modello
        if input("Vuoi cancellare anche i pesi del modello? (s/n): ").lower() == 's':
            import shutil
            folder_path = "Model"
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                print(f"Cartella {folder_path} cancellata")


# --- Esecuzione Principale ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())