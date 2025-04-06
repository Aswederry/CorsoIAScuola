import tkinter as tk

WINDOW_SIZE = 600
GRID_SIZE = 32
CELL_SIZE = 15

riceve = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
griglia = [[[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)] for _ in range(10)]
isTaken = [0 for _ in range(10)]

root = None
canvas = None
drawing = False


def create_main_window():
    """Crea la finestra principale con la griglia e i bottoni."""
    global root, canvas
    root = tk.Tk()
    root.title("Griglia 32x32")

    # Creazione del canvas
    canvas = tk.Canvas(root, width=WINDOW_SIZE, height=WINDOW_SIZE, bg="white")
    canvas.pack()

    # Creazione dei bottoni
    tk.Button(root, text="Inserisci", command=lambda: on_button_click(-1)).place(x=500, y=20, width=90, height=40)
    tk.Button(root, text="Cancella", command=lambda: on_button_click(10)).place(x=500, y=80, width=90, height=40)
    for i in range(10):
        tk.Button(root, text=str(i), command=lambda x=i: on_button_click(x)).place(x=8 + i * 48, y=500, width=40,
                                                                                   height=40)

    # Creazione delle linee della griglia principale
    create_grid_lines(canvas, "black")

    # Binding degli eventi
    root.bind("<Button-1>", on_button_press)
    root.bind("<ButtonRelease-1>", on_button_release)
    root.bind("<Motion>", on_move)
    root.resizable(False, False)


def create_grid_lines(canvas, color="black"):
    """Crea le linee della griglia su un canvas con un colore specificato."""
    for i in range(GRID_SIZE):
        x = i * CELL_SIZE
        y = i * CELL_SIZE
        canvas.create_line(x, 0, x, WINDOW_SIZE - 120, fill=color)
        canvas.create_line(0, y, WINDOW_SIZE - 120, y, fill=color)
    canvas.create_line(480, 0, 480, WINDOW_SIZE - 120, fill=color)
    canvas.create_line(0, 480, WINDOW_SIZE - 120, 480, fill=color)


def fillGrid(Cv, matrix):
    """Riempie la griglia sul canvas Cv usando la matrice specificata."""
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            x1 = j * CELL_SIZE
            y1 = i * CELL_SIZE
            x2 = x1 + CELL_SIZE
            y2 = y1 + CELL_SIZE
            colore_rgba = (int(matrix[i][j] * 255), 0, 0, 1)
            Cv.create_rectangle(x1, y1, x2, y2, fill=rgba_to_hex(colore_rgba))


def rgba_to_hex(rgba):
    """Converte un colore RGBA in formato esadecimale."""
    r, g, b, a = rgba
    hex_color = f"#{r:02x}{g:02x}{b:02x}"
    return hex_color


def OpenNewWindow(index):
    """Apre una nuova finestra e mostra la griglia salvata all'indice specificato."""
    newWindow = tk.Toplevel()
    newWindow.title(f"Griglia {index}")
    newCanvas = tk.Canvas(newWindow, width=WINDOW_SIZE - 120, height=WINDOW_SIZE - 120, bg="white")
    newCanvas.pack()
    create_grid_lines(newCanvas, "white")
    fillGrid(newCanvas, griglia[index])


def on_button_click(index):
    """Gestisce il click sui bottoni numerati, Inserisci e Cancella."""
    if index == 10:  # Bottone Cancella
        cancella()
    elif index == -1:  # Bottone Inserisci
        for i in range(10):
            if isTaken[i] == 0:
                griglia[i] = [row[:] for row in riceve]
                isTaken[i] = 1
                print(f"Salvato in griglia[{i}]")
                break
    elif 0 <= index < 10:  # Bottoni numerati
        isTaken[index] += 1
        griglia[index] = GetNewMatrix(riceve, griglia[index], index)
        cancella()
        OpenNewWindow(index)


def GetNewMatrix(m1, m2, index):
    """Calcola una nuova matrice combinando m1 e m2."""
    newMatrix = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            newMatrix[i][j] = (m1[i][j] + m2[i][j] * (isTaken[index] - 1)) / isTaken[index]
    return newMatrix


def cancella():
    """Pulisce la griglia principale."""
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            x1 = j * CELL_SIZE
            y1 = i * CELL_SIZE
            x2 = x1 + CELL_SIZE
            y2 = y1 + CELL_SIZE
            canvas.create_rectangle(x1, y1, x2, y2, fill="white")
            riceve[i][j] = 0


def on_move(event):
    """Gestisce il movimento del mouse per disegnare sulla griglia."""
    global drawing
    if drawing and event.x < 480 and event.y < 480:
        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE
        x1 = col * CELL_SIZE
        y1 = row * CELL_SIZE
        x2 = x1 + CELL_SIZE
        y2 = y1 + CELL_SIZE
        canvas.create_rectangle(x1, y1, x2, y2, fill="red")
        riceve[row][col] = 1


def on_button_press(event):
    """Inizia a disegnare quando il pulsante del mouse è premuto."""
    global drawing
    drawing = True


def on_button_release(event):
    """Ferma il disegno quando il pulsante del mouse è rilasciato."""
    global drawing
    drawing = False


if __name__ == "__main__":
    create_main_window()
    root.mainloop()