from datasets import load_dataset

import SavingAndLoading
from Model import Model  # Assumo che Model sia nel file Model.py
import numpy as np


def train_model(dataset, input_shape=(32, 32), num_neurons=100, num_outputs=10, learning_rate=0.001,
                max_epochs=10, batch_size=32, patience=5):
    """
    Addestra un modello sul dataset fornito.

    Parametri:
        dataset: dataset Hugging Face con chiavi 'image' e 'label'
        input_shape: tupla (righe, colonne) della forma dell'input
        num_neurons: numero di neuroni negli strati nascosti
        num_outputs: numero di classi in output
        learning_rate: tasso di apprendimento
        max_epochs: numero massimo di epoche
        batch_size: dimensione del batch
        patience: numero di epoche senza miglioramento per early stopping

    Restituisce:
        tuple (model, losses), modello addestrato e lista delle perdite per epoca
    """
    # Crea il modello
    model = Model(input_shape=input_shape, neuronsN=num_neurons, outputsN=num_outputs, learning_rate=learning_rate)

    # Prepara i dati
    n_samples = len(dataset)
    inputs = np.zeros((n_samples, input_shape[0], input_shape[1]))
    labels = np.zeros(n_samples, dtype=int)

    print("Preparazione dei dati...")
    for i in range(n_samples):
        # Ridimensiona l'immagine a (32, 32) e converte in array NumPy
        img = dataset[i]['image'].resize(input_shape)
        img_array = np.array(img, dtype=float) / 255.0  # Normalizza in [0, 1]
        inputs[i] = img_array
        labels[i] = dataset[i]['label']

    print("Inizio del training...")
    # Addestra il modello usando il metodo train del nuovo Model
    losses = model.train(inputs, labels, max_epochs=max_epochs, batch_size=batch_size, patience=patience)

    return model, losses


if __name__ == "__main__":
    # Carica il dataset MNIST (split di training)
    ds = load_dataset("ylecun/mnist", split="train")

    # Visualizza informazioni su un esempio
    img = ds[0]['image']
    print("Immagine PIL:", img)
    converted_array = np.array(img)
    print("Array NumPy shape:", converted_array.shape)

    # Addestra il modello
    model, losses = train_model(
        ds,
        input_shape=(32, 32),
        num_neurons=100,
        num_outputs=10,
        learning_rate=0.05,
        max_epochs=50,
        batch_size=32,
        patience=5
    )

    # --- Test su un esempio ---
    print("\n--- Test su un esempio ---")
    test_img = ds[0]['image'].resize((32, 32))
    test_img_array = np.array(test_img, dtype=float) / 255.0  # Normalizza
    test_img_batch = test_img_array[np.newaxis, ...]  # Aggiunge dimensione batch
    test_label = ds[0]['label']
    predictions = model.predict(test_img_batch)[0]  # Prende il primo elemento del batch
    predicted_class = np.argmax(predictions)
    print(f"Predizioni (probabilità): {predictions}")
    print(f"Classe predetta: {predicted_class}, Etichetta vera: {test_label}")

    # Salva il modello
    SavingAndLoading.save_new_model(model)
