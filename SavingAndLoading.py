import os
import numpy as np


def save_weights_and_inserts(matrix, inserts, weights_file, inserts_file):
    """
    Salva pesi e inserti (es. bias) su disco come file NumPy.

    Parametri:
        matrix: array NumPy contenente i pesi
        inserts: array NumPy contenente gli inserti (es. bias)
        weights_file: percorso del file per salvare i pesi
        inserts_file: percorso del file per salvare gli inserti
    """
    try:
        weights = np.array(matrix)
        inserts = np.array(inserts)
        np.save(weights_file, weights)
        np.save(inserts_file, inserts)
        print(f"Pesi salvati in {weights_file}, inserti salvati in {inserts_file}")
    except Exception as e:
        print(f"Errore durante il salvataggio: {e}")


def load_weights_and_inserts(weights_file, inserts_file):
    """
    Carica pesi e inserti da file NumPy.

    Parametri:
        weights_file: percorso del file dei pesi
        inserts_file: percorso del file degli inserti

    Restituisce:
        tuple (weights, inserts), liste dei pesi e inserti caricati, o None se i file non esistono
    """
    try:
        if not os.path.exists(weights_file):
            print(f"Errore: il file dei pesi {weights_file} non esiste")
            return None, None

        if not os.path.exists(inserts_file):
            print(f"Errore: il file degli inserti {inserts_file} non esiste")
            return None, None

        weights = np.load(weights_file).tolist()
        inserts = np.load(inserts_file).tolist()
        print(f"Pesi caricati da {weights_file}, inserti caricati da {inserts_file}")
        return weights, inserts

    except Exception as e:
        print(f"Errore durante il caricamento: {e}")
        return None, None


def delete_weights(file):
    """
    Elimina un file di pesi dopo conferma dell'utente.

    Parametri:
        file: percorso del file da eliminare
    """
    try:
        if not os.path.exists(file):
            print(f"Errore: il file {file} non esiste")
            return

        conferma = input(
            "ATTEN button_questo cancellerà il file specificato!\nScrivi 'Fortnite' per continuare: "
        )
        if conferma == "Fortnite":
            os.remove(file)
            print(f"File {file} cancellato con successo")
        else:
            print("Cancellazione annullata")
    except Exception as e:
        print(f"Errore durante la cancellazione: {e}")


def save_new_model(model, folder_path="Model"):
    """
    Salva i pesi e i bias del modello in una cartella.

    Parametri:
        model: istanza del modello neurale (classe Model)
        folder_path: percorso della cartella dove salvare i file
    """
    try:
        # Crea la cartella se non esiste
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Cartella creata: {folder_path}")

        # Salva ogni parametro del modello
        np.save(os.path.join(folder_path, "W1.npy"), model.W1)
        np.save(os.path.join(folder_path, "b1.npy"), model.b1)
        np.save(os.path.join(folder_path, "W2.npy"), model.W2)
        np.save(os.path.join(folder_path, "b2.npy"), model.b2)
        np.save(os.path.join(folder_path, "W_out.npy"), model.W_out)
        np.save(os.path.join(folder_path, "b_out.npy"), model.b_out)
        print(f"Modello salvato nella cartella: {folder_path}")

    except Exception as e:
        print(f"Errore durante il salvataggio del modello: {e}")


def load_new_model(model, folder_path="Model"):
    """
    Carica i pesi e i bias del modello da una cartella.

    Parametri:
        model: istanza del modello neurale (classe Model)
        folder_path: percorso della cartella da cui caricare i file

    Restituisce:
        model: modello con pesi e bias aggiornati
    """
    try:
        # Verifica che la cartella esista
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"La cartella {folder_path} non esiste")

        # Verifica che tutti i file necessari esistano
        required_files = ["W1.npy", "b1.npy", "W2.npy", "b2.npy", "W_out.npy", "b_out.npy"]
        for f in required_files:
            if not os.path.exists(os.path.join(folder_path, f)):
                raise FileNotFoundError(f"Il file {f} non esiste in {folder_path}")

        # Carica i parametri
        model.W1 = np.load(os.path.join(folder_path, "W1.npy"))
        model.b1 = np.load(os.path.join(folder_path, "b1.npy"))
        model.W2 = np.load(os.path.join(folder_path, "W2.npy"))
        model.b2 = np.load(os.path.join(folder_path, "b2.npy"))
        model.W_out = np.load(os.path.join(folder_path, "W_out.npy"))
        model.b_out = np.load(os.path.join(folder_path, "b_out.npy"))
        print(f"Modello caricato dalla cartella: {folder_path}")
        return model

    except Exception as e:
        print(f"Errore durante il caricamento del modello: {e}")
        return None


# --- Esempio di utilizzo ---
if __name__ == "__main__":
    from Model import Model  # Assumo che Model sia definito in Model.py

    # Crea un modello di esempio
    input_shape = (32, 32)
    model = Model(input_shape=input_shape, neuronsN=100, outputsN=10, learning_rate=0.001)

    # Salva il modello
    save_new_model(model, folder_path="TestModel")

    # Carica il modello
    loaded_model = load_new_model(model, folder_path="TestModel")

    # Esempio di cancellazione di un file di pesi
    delete_weights(os.path.join("TestModel", "W1.npy"))

    # Esempio di salvataggio e caricamento generico di pesi e inserti
    dummy_weights = np.random.randn(10, 10)
    dummy_inserts = np.zeros(10)
    save_weights_and_inserts(dummy_weights, dummy_inserts, "dummy_weights.npy", "dummy_inserts.npy")
    weights, inserts = load_weights_and_inserts("dummy_weights.npy", "dummy_inserts.npy")
    if weights is not None and inserts is not None:
        print("Pesi e inserti caricati con successo")