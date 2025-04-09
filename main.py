import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QMainWindow
)
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QMouseEvent, QPaintEvent
from PySide6.QtCore import Qt, QPoint, QRect, QSize
import functools

# --- Costanti ---
gridSize = 32
cellSize = 15
mainGridPixelSize = gridSize * cellSize
# Costanti per il layout della finestra
mainGridX = 10
mainGridY = 10
buttonAreaY = mainGridY + mainGridPixelSize + 20
labelAreaY1 = buttonAreaY + 50
labelAreaY2 = buttonAreaY + 70
rightPanelX = mainGridX + mainGridPixelSize + 20
weightGridCellSize = 3
weightGridPixelSize = gridSize * weightGridCellSize
weightGridSpacing = 15
windowWidth = rightPanelX + 2 * (weightGridPixelSize + weightGridSpacing + 100)
windowHeight = mainGridY + mainGridPixelSize + 120

# --- Dati Globali ---
# 'riceve' contiene i dati della griglia attualmente disegnata
riceve = [[0 for _ in range(gridSize)] for _ in range(gridSize)]
# 'griglia' contiene le matrici dei pesi appresi per ogni cifra (0-9)
griglia = [[[0.0 for _ in range(gridSize)] for _ in range(gridSize)] for _ in range(10)]
# 'isTaken' conta quante volte ogni cifra è stata cliccata
isTaken = [0 for _ in range(10)]
# 'sums' memorizza una somma calcolata relativa al riconoscimento
sums = [0.0 for _ in range(10)]


# --- Widget di Disegno per la Griglia Principale ---
class DrawingGrid(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(mainGridPixelSize, mainGridPixelSize)
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
        for r in range(gridSize):
            for c in range(gridSize):
                if riceve[r][c] == 1:
                    rect = QRect(c * cellSize, r * cellSize, cellSize, cellSize)
                    painter.fillRect(rect, Qt.black)

        # Disegna le linee della griglia
        painter.setPen(QPen(Qt.lightGray, 1))  # Linee della griglia più chiare
        for i in range(gridSize + 1):
            x = i * cellSize
            y = i * cellSize
            painter.drawLine(x, 0, x, mainGridPixelSize)  # Verticale
            painter.drawLine(0, y, mainGridPixelSize, y)  # Orizzontale

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
        c = pos.x() // cellSize
        r = pos.y() // cellSize
        if 0 <= r < gridSize and 0 <= c < gridSize:
            if riceve[r][c] == 0:  # Aggiorna solo se cambiato per evitare di ridisegnare (così non lagga)
                riceve[r][c] = 1
                self.update()

    def ClearGrid(self):
        global riceve
        changed = False
        for r in range(gridSize):
            for c in range(gridSize):
                if riceve[r][c] != 0:
                    riceve[r][c] = 0
                    changed = True
        if changed:
            self.update()


# --- Widget per visualizzare una singola Griglia dei Pesi appresa ---
class WeightGridDisplay(QWidget):
    def __init__(self, index, parent=None):
        super().__init__(parent)
        self.index = index
        self.setFixedSize(weightGridPixelSize, weightGridPixelSize)
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(p)

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setPen(Qt.NoPen)

        matrix = griglia[self.index]
        for r in range(gridSize):
            for c in range(gridSize):
                # Limita il valore tra 0 e 1 prima di scalarlo a 0-255
                value = max(0.0, min(1.0, matrix[r][c]))
                gray = int(value * 255)
                color = QColor(gray, gray, gray)
                rect = QRect(c * weightGridCellSize, r * weightGridCellSize,
                             weightGridCellSize, weightGridCellSize)
                painter.fillRect(rect, color)

    def UpdateDisplay(self):
        self.update()


# --- Finestra Principale dell'Applicazione ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning")
        self.setFixedSize(windowWidth, windowHeight)

        # --- Widget Centrale per contenere tutto ---
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # --- Crea la Griglia di Disegno ---
        self.drawing_grid = DrawingGrid(central_widget)
        self.drawing_grid.move(mainGridX, mainGridY)

        # --- Crea i Pulsanti ---
        self.recognize_button = QPushButton("Riconosci", central_widget)
        self.recognize_button.setGeometry(rightPanelX, mainGridY + 20, 90, 40)
        self.recognize_button.clicked.connect(self._HandleRecognize)  # Logica per il pulsante Riconosci

        self.clear_button = QPushButton("Cancella", central_widget)
        self.clear_button.setGeometry(rightPanelX, mainGridY + 80, 90, 40)
        self.clear_button.clicked.connect(self._HandleClear)  # Logica per il pulsante Cancella

        self.number_buttons = []
        for i in range(10):
            button = QPushButton(str(i), central_widget)
            button.setGeometry(mainGridX + i * 48, buttonAreaY, 40, 40)
            button.clicked.connect(functools.partial(self._HandleNumberButton, i))
            self.number_buttons.append(button)

        # --- Crea i label ---
        self.isTakenLabels = []
        self.sumLabels = []
        for i in range(10):
            # label isTaken
            label_taken = QLabel("0", central_widget)
            label_taken.setGeometry(mainGridX + i * 48 + 10, labelAreaY1, 30, 20)
            label_taken.setAlignment(Qt.AlignCenter)
            self.isTakenLabels.append(label_taken)

            # label Somma
            label_sum = QLabel("0.0", central_widget)
            label_sum.setGeometry(mainGridX + i * 48 + 5, labelAreaY2, 40, 20)
            label_sum.setAlignment(Qt.AlignCenter)
            self.sumLabels.append(label_sum)

        # --- Crea i Visualizzatori delle Griglie dei Pesi ---
        self.weight_grid_displays = []
        for i in range(10):
            display = WeightGridDisplay(i, central_widget)
            # Disponi in 2 colonne da 5
            col = i // 5
            row = i % 5
            x_pos = 150 + rightPanelX + col * (weightGridPixelSize + weightGridSpacing)
            y_pos = 20 + mainGridY + row * (weightGridPixelSize + weightGridSpacing)
            display.move(x_pos, y_pos)
            self.weight_grid_displays.append(display)

        self._UpdateNumberInfo()

    # --- Gestori dei Pulsanti ---
    def _HandleRecognize(self):
        return  # Per ora non fa niente perchè non so come si fa, prossima lezione lo facciamo

    def _HandleClear(self):
        self.drawing_grid.ClearGrid()

    def _HandleNumberButton(self, index):
        isTaken[index] += 1
        griglia[index] = self._CalculateNewMatrix(riceve, griglia[index], index)
        sums[index] = self._CalculateSum(index)

        # Aggiorna il visualizzatore della griglia dei pesi corrispondente
        self.weight_grid_displays[index].UpdateDisplay()

        # Aggiorna le label
        self._UpdateNumberInfo()

        # Pulisci la griglia di disegno principale dopo l'elaborazione
        self.drawing_grid.ClearGrid()

    # --- Funzioni di Supporto ---
    def _CalculateNewMatrix(self, m1, m2, index):
        """Calcola una nuova matrice combinando m1 e m2 (media)."""
        newMatrix = [[0.0 for _ in range(gridSize)] for _ in range(gridSize)]
        count = isTaken[index]  # Il nuovo conteggio totale
        if count <= 0:
            return m2

        # Peso precedente = (conteggio - 1) / conteggio
        # Peso nuovo campione = 1 / conteggio
        prev_weight = (count - 1) / count
        new_weight = 1 / count

        for i in range(gridSize):
            for j in range(gridSize):
                # Media ponderata: (media_esistente * (n-1) + nuovo_valore) / n
                newMatrix[i][j] = (m2[i][j] * prev_weight) + (m1[i][j] * new_weight)
                newMatrix[i][j] = max(0.0, min(1.0, newMatrix[i][j]))
        return newMatrix

    def _CalculateSum(self, index):
        """Calcola la somma del prodotto scalare tra 'riceve' e 'griglia[index]'."""
        current_sum = 0.0
        weight_matrix = griglia[index]
        input_matrix = riceve

        for i in range(gridSize):
            for j in range(gridSize):
                current_sum += input_matrix[i][j] * weight_matrix[i][j]
        return current_sum

    def _UpdateNumberInfo(self):
        """Aggiorna il testo delle label isTaken e sum."""
        for i in range(10):
            self.isTakenLabels[i].setText(str(isTaken[i]))
            # Formatta la somma con poche cifre decimali per leggibilità
            self.sumLabels[i].setText(f"{sums[i]:.2f}")


# --- Esecuzione Principale ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
