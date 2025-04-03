import tkinter as tk

WINDOW_SIZE = 600
GRID_SIZE = 32
CELL_SIZE = 15

root = tk.Tk()
root.title("Griglia 32x32")

# Inizializzazione corretta delle matrici
riceve = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
griglia = [[[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)] for _ in range(10)]
isTaken = [0 for _ in range(10)]

canvas = tk.Canvas(root, width=WINDOW_SIZE, height=WINDOW_SIZE, bg="white")
canvas.pack()


def on_button_click(index):
    if index == 10:  # Gestione del bottone Cancella
        cancella()
    elif index == -1:  # Gestione del bottone Inserisci
        for i in range(10):
            if isTaken[i] == 0:
                griglia[i] = [row[:] for row in riceve]
                isTaken[i] = 1
                print(f"Salvato in griglia[{i}]")
                break
    elif 0 <= index < 10:
        if isTaken[index] == 0:
            griglia[index] = [row[:] for row in riceve]
            isTaken[index] = 1
            print(f"Salvato in griglia[{index}]")
        else:
            isTaken[index] += 1
            OpenNewWindow(index)


def OpenNewWindow(index):
    newWindow = tk.Toplevel()
    newWindow.title(f"Griglia {index}")
    newCanvas = tk.Canvas(newWindow, width=WINDOW_SIZE, height=WINDOW_SIZE, bg="white")
    newCanvas.pack()

    for i in range(GRID_SIZE):
        x = i * CELL_SIZE
        y = i * CELL_SIZE
        newCanvas.create_line(x, 0, x, WINDOW_SIZE - 120, fill="white")
        newCanvas.create_line(0, y, WINDOW_SIZE - 120, y, fill="white")
    newCanvas.create_line(480, 0, 480, WINDOW_SIZE - 120, fill="white")
    newCanvas.create_line(0, 480, WINDOW_SIZE - 120, 480, fill="white")

    fillGrid(newCanvas, index)


def fillGrid(Cv, index):
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            x1 = j * CELL_SIZE
            y1 = i * CELL_SIZE
            x2 = x1 + CELL_SIZE
            y2 = y1 + CELL_SIZE

            newMatrix = GetNewMatrix(riceve, griglia[index], index)

            colore_rgba = (int(newMatrix[i][j] * 255), 0, 0, 1)

            Cv.create_rectangle(x1, y1, x2, y2, fill=rgba_to_hex(colore_rgba))


def rgba_to_hex(rgba):
    r, g, b, a = rgba
    hex_color = f"#{r:02x}{g:02x}{b:02x}"
    return hex_color


def GetNewMatrix(m1, m2, index):
    newMatrix = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            #  newMatrix[i][j] = (m1[i][j] + m2[i][j]) / 2
            newMatrix[i][j] = (m1[i][j] + m2[i][j] * (isTaken[index] - 1)) / isTaken[index]
    return newMatrix


def cancella():
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            x1 = j * CELL_SIZE
            y1 = i * CELL_SIZE
            x2 = x1 + CELL_SIZE
            y2 = y1 + CELL_SIZE
            canvas.create_rectangle(x1, y1, x2, y2, fill="white")
            riceve[i][j] = 0


# Creazione bottoni
tk.Button(root, text="Inserisci", command=lambda: on_button_click(-1)).place(x=500, y=20, width=90, height=40)
tk.Button(root, text="Cancella", command=lambda: on_button_click(10)).place(x=500, y=80, width=90, height=40)

for i in range(10):
    tk.Button(root, text=str(i), command=lambda x=i: on_button_click(x)).place(x=8 + i * 48, y=500, width=40, height=40)

# Creazione griglia principale
for i in range(GRID_SIZE):
    x = i * CELL_SIZE
    y = i * CELL_SIZE
    canvas.create_line(x, 0, x, WINDOW_SIZE - 120, fill="black")
    canvas.create_line(0, y, WINDOW_SIZE - 120, y, fill="black")
canvas.create_line(480, 0, 480, WINDOW_SIZE - 120, fill="black")
canvas.create_line(0, 480, WINDOW_SIZE - 120, 480, fill="black")

drawing = False


def on_move(event):
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
    global drawing
    drawing = True


def on_button_release(event):
    global drawing
    drawing = False


root.bind("<Button-1>", on_button_press)
root.bind("<ButtonRelease-1>", on_button_release)
root.bind("<Motion>", on_move)
root.resizable(False, False)
root.mainloop()
