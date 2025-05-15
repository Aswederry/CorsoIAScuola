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
# Adjust WINDOW_WIDTH for horizontal displays. Start with a base.
WINDOW_WIDTH = 900  # Initial guess, will be dynamically adjusted.
WINDOW_HEIGHT = 700  # Might not need to be as tall now.

WEIGHTS_NAME = "weights.npy"
NUMS_NAME = "nums.npy"

MODEL = Model(input_shape=(GRID_SIZE, GRID_SIZE), neuronsN=100, outputsN=10, learning_rate=0.001)

RICEVE = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
GRIGLIA = [[[0.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)] for _ in range(10)]
IS_TAKEN = [0 for _ in range(10)]
SUMS = [0.0 for _ in range(10)]
PERCENTUALI = [0.0 for _ in range(10)]


class DrawingGrid(QWidget):
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
        painter.setPen(Qt.NoPen)
        painter.setBrush(Qt.black)
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if RICEVE[r][c] == 1:
                    rect = QRect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    painter.fillRect(rect, Qt.black)
        painter.setPen(QPen(Qt.lightGray, 1))
        for i in range(GRID_SIZE + 1):
            x = i * CELL_SIZE;
            y = i * CELL_SIZE
            painter.drawLine(x, 0, x, MAIN_GRID_PIXEL_SIZE)
            painter.drawLine(0, y, MAIN_GRID_PIXEL_SIZE, y)

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
        if 0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE and RICEVE[r][c] == 0:
            RICEVE[r][c] = 1
            self.update()

    def clear_grid(self):
        global RICEVE
        changed = any(RICEVE[r][c] != 0 for r in range(GRID_SIZE) for c in range(GRID_SIZE))
        if changed:
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    RICEVE[r][c] = 0
            self.update()


class WeightGridDisplay(QWidget):
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
        for r_idx in range(GRID_SIZE):
            for c_idx in range(GRID_SIZE):
                value = max(0.0, min(1.0, matrix[r_idx][c_idx]))
                gray = int(value * 255)
                color = QColor(gray, gray, gray)
                rect = QRect(c_idx * WEIGHT_GRID_CELL_SIZE, r_idx * WEIGHT_GRID_CELL_SIZE,
                             WEIGHT_GRID_CELL_SIZE, WEIGHT_GRID_CELL_SIZE)
                painter.fillRect(rect, color)

    def update_display(self):
        self.update()


class LittleBarsGrid(QWidget):
    BAR_MAX_WIDTH = 40
    BAR_HEIGHT = 2
    BAR_SPACING_Y = 1

    BAR_ACTIVE_COLOR = QColor(Qt.blue)
    BAR_BACKGROUND_COLOR = QColor(220, 220, 220)
    BAR_OUTPUT_ACTIVE_COLOR = QColor(Qt.red)

    def __init__(self, matrix_data=None, parent=None, is_output_layer=False):
        super().__init__(parent)
        self.matrix_values = None
        self.num_neurons = 0
        self.is_output_layer = is_output_layer
        self.min_val = 0.0
        self.max_val = 1.0
        self.range_val = 1.0

        self._process_matrix_data(matrix_data)

        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(p)

    def _process_matrix_data(self, matrix_data):
        if matrix_data is None or \
                (hasattr(matrix_data, 'size') and matrix_data.size == 0) or \
                (not hasattr(matrix_data, 'size') and isinstance(matrix_data, (list, tuple)) and len(matrix_data) == 0):
            self.matrix_values = np.array([0.0])
        else:
            temp_matrix = np.array(matrix_data, dtype=float).flatten()
            if temp_matrix.size == 0:
                self.matrix_values = np.array([0.0])
            else:
                self.matrix_values = temp_matrix

        self.num_neurons = self.matrix_values.shape[0]

        if self.num_neurons > 0:
            current_min = np.min(self.matrix_values)
            current_max = np.max(self.matrix_values)

            if self.is_output_layer:
                self.min_val = current_min
                self.max_val = current_max
                self.range_val = self.max_val - self.min_val
                if self.range_val == 0:
                    self.range_val = 1.0
            else:
                self.min_val = 0.0
                self.max_val = 1.0
                self.range_val = 1.0

        if self.num_neurons > 0:
            total_widget_width = self.BAR_MAX_WIDTH
            cell_total_height = self.BAR_HEIGHT + self.BAR_SPACING_Y
            total_widget_height = self.num_neurons * cell_total_height - self.BAR_SPACING_Y
        else:
            total_widget_width = self.BAR_MAX_WIDTH
            total_widget_height = self.BAR_HEIGHT

        self.setFixedSize(max(self.BAR_MAX_WIDTH, total_widget_width),
                          max(self.BAR_HEIGHT, total_widget_height))

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        if self.matrix_values is None or self.num_neurons == 0:
            return

        cell_box_height_with_spacing = self.BAR_HEIGHT + self.BAR_SPACING_Y
        active_color = self.BAR_OUTPUT_ACTIVE_COLOR if self.is_output_layer else self.BAR_ACTIVE_COLOR

        for r in range(self.num_neurons):
            raw_value = self.matrix_values[r]
            normalized_value = (raw_value - self.min_val) / self.range_val if self.range_val != 0 else 0.0
            normalized_value = max(0.0, min(1.0, normalized_value))

            current_bar_width = int(normalized_value * self.BAR_MAX_WIDTH)
            x_cell_start = 0
            y_cell_start = r * cell_box_height_with_spacing

            painter.fillRect(x_cell_start, y_cell_start,
                             self.BAR_MAX_WIDTH, self.BAR_HEIGHT,
                             self.BAR_BACKGROUND_COLOR)

            if current_bar_width > 0:
                painter.fillRect(x_cell_start, y_cell_start,
                                 current_bar_width, self.BAR_HEIGHT,
                                 active_color)

    def update_display(self, new_matrix_data=None):
        self._process_matrix_data(new_matrix_data)
        self.update()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Riconoscimento Cifre")
        self.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.drawing_grid = DrawingGrid(self.central_widget)
        self.drawing_grid.move(MAIN_GRID_X, MAIN_GRID_Y)

        self.recognize_button = QPushButton("Riconosci", self.central_widget)
        self.recognize_button.setGeometry(RIGHT_PANEL_X, MAIN_GRID_Y + 20, 110, 30)
        self.recognize_button.clicked.connect(self._handle_recognize)

        self.clear_button = QPushButton("Cancella Disegno", self.central_widget)
        self.clear_button.setGeometry(RIGHT_PANEL_X, MAIN_GRID_Y + 60, 110, 30)
        self.clear_button.clicked.connect(self._handle_clear)

        self.save_weights_button = QPushButton("Salva Pesi & Modello", self.central_widget)
        self.save_weights_button.setGeometry(RIGHT_PANEL_X, MAIN_GRID_Y + 100, 110, 30)
        self.save_weights_button.clicked.connect(self._save_weights_button)

        self.load_weights_button = QPushButton("Carica Pesi & Modello", self.central_widget)
        self.load_weights_button.setGeometry(RIGHT_PANEL_X, MAIN_GRID_Y + 140, 110, 30)
        self.load_weights_button.clicked.connect(self._load_weights_button)

        self.delete_weights_button = QPushButton("Cancella Dati Salvati", self.central_widget)
        self.delete_weights_button.setGeometry(RIGHT_PANEL_X, MAIN_GRID_Y + 180, 110, 30)
        self.delete_weights_button.clicked.connect(self._delete_weights_button)

        self.number_buttons = []
        for i in range(10):
            button = QPushButton(str(i), self.central_widget)
            button.setGeometry(MAIN_GRID_X + i * 48, BUTTON_AREA_Y, 40, 40)
            button.clicked.connect(functools.partial(self._handle_number_button, i))
            self.number_buttons.append(button)

        self.is_taken_labels, self.sum_labels, self.recognition_percent = [], [], []
        for i in range(10):
            label_taken = QLabel("0", self.central_widget)
            label_taken.setGeometry(MAIN_GRID_X + i * 48 + 10, LABEL_AREA_Y1, 30, 20)
            label_taken.setAlignment(Qt.AlignCenter);
            self.is_taken_labels.append(label_taken)
            label_sum = QLabel("0.0", self.central_widget)
            label_sum.setGeometry(MAIN_GRID_X + i * 48 + 5, LABEL_AREA_Y2, 40, 20)
            label_sum.setAlignment(Qt.AlignCenter);
            self.sum_labels.append(label_sum)
            label_percent = QLabel("0.00%", self.central_widget)
            label_percent.setGeometry(MAIN_GRID_X + i * 48 + 5, LABEL_AREA_Y3, 40, 20)
            label_percent.setAlignment(Qt.AlignCenter);
            self.recognition_percent.append(label_percent)

        self.weight_grid_displays = []
        wg_display_start_x = RIGHT_PANEL_X + 110 + 20
        for i in range(10):
            display = WeightGridDisplay(i, self.central_widget)
            col, row = i // 5, i % 5
            x_pos = wg_display_start_x + col * (WEIGHT_GRID_PIXEL_SIZE + WEIGHT_GRID_SPACING)
            y_pos = MAIN_GRID_Y + 20 + row * (WEIGHT_GRID_PIXEL_SIZE + WEIGHT_GRID_SPACING)
            display.move(x_pos, y_pos);
            self.weight_grid_displays.append(display)

        self._update_number_info()

        self.hidden_bars_display_widget = None
        self.hidden_layer_label = None
        self.output_bars_display_widget = None
        self.output_layer_label = None

    def _handle_recognize(self):
        global PERCENTUALI, RICEVE, GRIGLIA, MODEL
        input_drawing_flat = np.array(RICEVE).astype(float)
        active_pixels_count = np.sum(input_drawing_flat)
        for i in range(10):
            PERCENTUALI[i] = (np.sum(input_drawing_flat * np.array(
                GRIGLIA[i])) / active_pixels_count * 100.0) if active_pixels_count > 0 else 0.0
            self.recognition_percent[i].setText(f"{PERCENTUALI[i]:.2f}%")

        input_array = np.array(RICEVE, dtype=float)
        input_batch = input_array[np.newaxis, ...]
        predictions_raw_output_layer = MODEL.predict(input_batch)

        if predictions_raw_output_layer is not None:
            predictions_percent = predictions_raw_output_layer[0] * 100
            predicted_class = np.argmax(predictions_percent)
            print(f"Predizioni modello (output layer, %): {['{:.2f}'.format(p) for p in predictions_percent]}")
            print(f"Classe predetta dal modello: {predicted_class}")
        else:
            print("Predizione del modello non disponibile.")
            return

        # --- Horizontal Layout for Bar Displays ---
        # Start Y position for both bar displays (below the delete button)
        bars_y_start_for_labels = MAIN_GRID_Y + 180 + 30 + 10
        bars_y_start_for_grids = bars_y_start_for_labels + 18 + 2  # Label height + spacing

        # Start X for the first bar display (a1)
        current_x_offset = RIGHT_PANEL_X
        max_height_of_bars_row = 0  # To track max height in this row for window resize

        # --- Display Hidden Layer (a1) Activations ---
        if hasattr(MODEL, 'a1') and MODEL.a1 is not None:
            activations_a1 = MODEL.a1[0]

            if self.hidden_bars_display_widget is None:
                self.hidden_bars_display_widget = LittleBarsGrid(activations_a1, self.central_widget,
                                                                 is_output_layer=False)
                self.hidden_layer_label = QLabel("Strato A1:", self.central_widget)
                # Label width can be dynamic or fixed. For simplicity, use widget width.
                self.hidden_layer_label.setFixedWidth(self.hidden_bars_display_widget.width())

            self.hidden_layer_label.move(current_x_offset, bars_y_start_for_labels)
            self.hidden_bars_display_widget.update_display(activations_a1)  # Update first to get correct size
            self.hidden_bars_display_widget.move(current_x_offset, bars_y_start_for_grids)

            self.hidden_bars_display_widget.show()
            self.hidden_layer_label.show()

            max_height_of_bars_row = max(max_height_of_bars_row, self.hidden_bars_display_widget.height())
            current_x_offset += self.hidden_bars_display_widget.width() + 20  # Add width + spacing for next display
        else:
            if self.hidden_bars_display_widget: self.hidden_bars_display_widget.hide()
            if self.hidden_layer_label: self.hidden_layer_label.hide()
            print("MODEL.a1 non disponibile per display.")

        # --- Display Output Layer (a2 or z_out) Activations ---
        data_for_output_display = None
        output_label_text = "Strato A2:"
        if hasattr(MODEL, 'a2') and MODEL.a2 is not None:
            data_for_output_display = MODEL.a2[0]
            output_label_text = "Output (a2 - logits):"
            print(f"MODEL.a2 (logits): {['{:.2f}'.format(x) for x in data_for_output_display]}")
        elif predictions_raw_output_layer is not None:
            data_for_output_display = predictions_raw_output_layer[0]
            output_label_text = "Output (y_hat - probabilità):"
            print(f"MODEL y_hat (probabilities): {['{:.2f}'.format(x) for x in data_for_output_display]}")

        if data_for_output_display is not None:
            if self.output_bars_display_widget is None:
                self.output_bars_display_widget = LittleBarsGrid(data_for_output_display, self.central_widget,
                                                                 is_output_layer=True)
                self.output_layer_label = QLabel(output_label_text, self.central_widget)
                self.output_layer_label.setFixedWidth(self.output_bars_display_widget.width())  # Match bar width
            else:
                self.output_layer_label.setText(output_label_text)  # Update text if source changed

            self.output_layer_label.move(current_x_offset, bars_y_start_for_labels)
            self.output_bars_display_widget.update_display(data_for_output_display)  # Update first for size
            self.output_bars_display_widget.move(current_x_offset, bars_y_start_for_grids)

            self.output_bars_display_widget.show()
            self.output_layer_label.show()

            max_height_of_bars_row = max(max_height_of_bars_row, self.output_bars_display_widget.height())
            current_x_offset += self.output_bars_display_widget.width() + 10  # End of this row
        else:
            if self.output_bars_display_widget: self.output_bars_display_widget.hide()
            if self.output_layer_label: self.output_layer_label.hide()
            print("Dati layer output non disponibili per display.")

        # --- Dynamic Window Resizing ---
        # Required height is based on the y_start of bars + max height in that row
        required_height = bars_y_start_for_grids + max_height_of_bars_row + 10
        # Required width is now current_x_offset (if bars are the rightmost) OR the template displays
        required_width_for_bars = current_x_offset
        required_width_for_templates = self.weight_grid_displays[-1].x() + self.weight_grid_displays[
            -1].width() + 10 if self.weight_grid_displays else 0

        final_required_width = max(required_width_for_bars, required_width_for_templates,
                                   self.drawing_grid.x() + self.drawing_grid.width() + RIGHT_PANEL_X + 110 + 20)  # Ensure main grid + buttons also fit

        new_height = self.height()
        if required_height > self.height():
            new_height = required_height

        new_width = self.width()
        if final_required_width > self.width():
            new_width = final_required_width

        if new_height != self.height() or new_width != self.width():
            print(f"Adjusting window size. W: {self.width()}->{new_width}, H: {self.height()}->{new_height}")
            self.setFixedSize(new_width, new_height)

    def _handle_clear(self):
        self.drawing_grid.clear_grid()

    def _handle_number_button(self, index):
        global IS_TAKEN, GRIGLIA, SUMS, MODEL, RICEVE
        IS_TAKEN[index] += 1
        GRIGLIA[index] = self._calculate_new_matrix(RICEVE, GRIGLIA[index], index)
        SUMS[index] = self._calculate_sum(index)
        input_array = np.array(RICEVE, dtype=float)
        input_batch = input_array[np.newaxis, ...]
        labels = np.array([index], dtype=int)
        loss = MODEL.train(input_batch, labels, max_epochs=1, batch_size=1, patience=1)
        print(f"Addestramento per cifra {index}, perdita: {loss[-1]:.4f}")
        self.weight_grid_displays[index].update_display()
        self._update_number_info()
        self.drawing_grid.clear_grid()

    def _calculate_new_matrix(self, m1, m2, index):
        new_matrix = [[0.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        count = IS_TAKEN[index]
        prev_weight = (count - 1) / count if count > 0 else 0
        new_weight = 1 / count if count > 0 else 1
        for r_idx in range(GRID_SIZE):
            for c_idx in range(GRID_SIZE):
                new_val = (m2[r_idx][c_idx] * prev_weight) + (m1[r_idx][c_idx] * new_weight)
                new_matrix[r_idx][c_idx] = max(0.0, min(1.0, new_val))
        return new_matrix

    def _calculate_sum(self, index):
        return np.sum(np.array(RICEVE) * np.array(GRIGLIA[index]))

    def _update_number_info(self):
        for i in range(10):
            self.is_taken_labels[i].setText(str(IS_TAKEN[i]))
            self.sum_labels[i].setText(f"{SUMS[i]:.2f}")

    def _save_weights_button(self):
        sal.save_weights_and_inserts(GRIGLIA, IS_TAKEN, WEIGHTS_NAME, NUMS_NAME)
        sal.save_new_model(MODEL, folder_path="Model")
        print("Pesi (GRIGLIA, IS_TAKEN) e modello neurale salvati.")

    def _load_weights_button(self):
        global GRIGLIA, IS_TAKEN, MODEL
        weights, inserts = sal.load_weights_and_inserts(WEIGHTS_NAME, NUMS_NAME)
        if weights is not None and inserts is not None:
            GRIGLIA, IS_TAKEN = weights, inserts
            for i in range(10): self.weight_grid_displays[i].update_display()
            self._update_number_info()
            print("Pesi (GRIGLIA, IS_TAKEN) caricati.")
        loaded_model = sal.load_new_model(MODEL, folder_path="Model")
        if loaded_model: MODEL = loaded_model; print("Modello neurale caricato.")

    def _delete_weights_button(self):
        sal.delete_weights(WEIGHTS_NAME, NUMS_NAME)
        print(f"File {WEIGHTS_NAME} e {NUMS_NAME} cancellati.")
        global GRIGLIA, IS_TAKEN, SUMS, PERCENTUALI
        GRIGLIA = [[[0.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)] for _ in range(10)]
        IS_TAKEN = [0 for _ in range(10)];
        SUMS = [0.0 for _ in range(10)];
        PERCENTUALI = [0.0 for _ in range(10)]
        for i in range(10):
            self.weight_grid_displays[i].update_display()
            self.recognition_percent[i].setText("0.00%")
        self._update_number_info()
        if input("Vuoi cancellare anche i pesi del modello neurale salvati? (s/n): ").lower() == 's':
            import shutil
            folder_path = "Model"
            if os.path.exists(folder_path):
                try:
                    shutil.rmtree(folder_path); print(f"Cartella '{folder_path}' cancellata.")
                except Exception as e:
                    print(f"Errore cancellazione '{folder_path}': {e}")
            else:
                print(f"Cartella '{folder_path}' non trovata.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())