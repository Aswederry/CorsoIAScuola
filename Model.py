import numpy as np
import matplotlib.pyplot as plt


# --- Funzioni di attivazione ---
def relu(x):
    """Funzione di attivazione ReLU."""
    return np.maximum(0, x)


def relu_derivative(x):
    """Derivata della funzione ReLU."""
    return (x > 0).astype(float)


def softmax(x):
    """Funzione di attivazione Softmax (stabile numericamente)."""
    # Sottrae il massimo per stabilità numerica
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# --- Funzione di perdita ---
def cross_entropy_loss(y_pred_probs, y_true_indices):
    """
    Calcola la perdita di Cross-Entropy per un batch.

    Parametri:
        y_pred_probs: array di forma (batch_size, outputsN), probabilità previste
        y_true_indices: array di forma (batch_size,), indici delle classi vere

    Restituisce:
        float, perdita media sul batch
    """
    batch_size = y_pred_probs.shape[0]
    # Estrae le probabilità per le classi vere
    correct_probs = y_pred_probs[np.arange(batch_size), y_true_indices]
    # Calcola -log(prob) con epsilon per stabilità
    correct_logprobs = -np.log(correct_probs + 1e-9)
    return np.mean(correct_logprobs)


class Model:
    def __init__(self, input_shape=(32, 32), neuronsN=100, outputsN=10, learning_rate=0.001):
        """
        Inizializza il modello di rete neurale.

        Parametri:
            input_shape: tupla (righe, colonne) della forma dell'input
            neuronsN: numero di neuroni negli strati nascosti
            outputsN: numero di classi in output
            learning_rate: tasso di apprendimento
        """
        self.input_rows, self.input_cols = input_shape
        self.input_size_flat = self.input_rows * self.input_cols  # Dimensione input appiattito
        self.neuronsN = neuronsN
        self.outputsN = outputsN
        self.learning_rate = learning_rate

        # --- Inizializzazione pesi e bias con He initialization ---
        # Strato 1 (Input -> Hidden 1)
        self.W1 = np.random.randn(neuronsN, self.input_size_flat) * np.sqrt(2.0 / self.input_size_flat)
        self.b1 = np.random.randn(neuronsN)

        # Strato 2 (Hidden 1 -> Hidden 2)
        self.W2 = np.random.randn(neuronsN, neuronsN) * np.sqrt(2.0 / neuronsN)
        self.b2 = np.random.randn(neuronsN)

        # Strato 3 (Hidden 2 -> Output)
        self.W_out = np.random.randn(outputsN, neuronsN) * np.sqrt(2.0 / neuronsN)
        self.b_out = np.random.randn(outputsN)

        # --- Placeholder per valori intermedi ---
        self.input_batch = None
        self.z1 = None
        self.z2 = None
        self.z_out = None
        self.a1 = None
        self.a2 = None
        self.probs = None

    def _forward(self, input_batch):
        """
        Esegue il passaggio forward attraverso la rete per un batch.

        Parametri:
            input_batch: array di forma (batch_size, input_rows, input_cols)

        Restituisce:
            array di forma (batch_size, outputsN), probabilità previste
        """
        # Controlla la forma dell'input
        if input_batch.shape[1:] != (self.input_rows, self.input_cols):
            raise ValueError(
                f"Forma input errata. Attesa (batch_size, {self.input_rows}, {self.input_cols}), "
                f"ricevuta {input_batch.shape}"
            )
        self.input_batch = input_batch  # Memorizza input

        # Appiattisce l'input: (batch_size, input_rows, input_cols) -> (batch_size, input_size_flat)
        input_flat = input_batch.reshape(-1, self.input_size_flat)

        # --- Strato 1: Input -> Hidden 1 ---
        self.z1 = np.dot(input_flat, self.W1.T) + self.b1  # (batch_size, neuronsN)
        self.a1 = relu(self.z1)

        # --- Strato 2: Hidden 1 -> Hidden 2 ---
        self.z2 = np.dot(self.a1, self.W2.T) + self.b2  # (batch_size, neuronsN)
        self.a2 = relu(self.z2)

        # --- Strato 3: Hidden 2 -> Output ---
        self.z_out = np.dot(self.a2, self.W_out.T) + self.b_out  # (batch_size, outputsN)
        self.probs = softmax(self.z_out)

        return self.probs

    def predict(self, input_batch):
        """
        Predice le probabilità per un batch di input.

        Parametri:
            input_batch: array di forma (batch_size, input_rows, input_cols)

        Restituisce:
            array di forma (batch_size, outputsN), probabilità previste
        """
        input_batch_np = np.array(input_batch)  # Assicura array NumPy
        probabilities = self._forward(input_batch_np)
        return probabilities

    def _backward(self, y_true_indices):
        """
        Esegue il passaggio backward (backpropagation) per calcolare i gradienti.

        Parametri:
            y_true_indices: array di forma (batch_size,), indici delle classi vere

        Restituisce:
            tuple dei gradienti (dW1, db1, dW2, db2, dW_out, db_out)
        """
        batch_size = self.input_batch.shape[0]

        # Crea vettore one-hot per le etichette vere
        y_true = np.zeros((batch_size, self.outputsN))
        y_true[np.arange(batch_size), y_true_indices] = 1

        # --- Gradienti per lo strato di output ---
        # Errore allo strato di output: dL/dz_out = probs - y_true
        delta_out = self.probs - y_true  # (batch_size, outputsN)

        # Gradiente di W_out: dL/dW_out = delta_out^T * a2
        dW_out = np.dot(delta_out.T, self.a2) / batch_size  # (outputsN, neuronsN)

        # Gradiente di b_out: media di delta_out lungo il batch
        db_out = np.mean(delta_out, axis=0)  # (outputsN,)

        # --- Gradienti per lo strato nascosto 2 ---
        # Errore propagato: dL/da2 = delta_out * W_out
        error_h2 = np.dot(delta_out, self.W_out)  # (batch_size, neuronsN)

        # Errore pre-attivazione: dL/dz2 = error_h2 * relu'(z2)
        delta_2 = error_h2 * relu_derivative(self.z2)  # (batch_size, neuronsN)

        # Gradiente di W2: dL/dW2 = delta_2^T * a1
        dW2 = np.dot(delta_2.T, self.a1) / batch_size  # (neuronsN, neuronsN)

        # Gradiente di b2: media di delta_2 lungo il batch
        db2 = np.mean(delta_2, axis=0)  # (neuronsN,)

        # --- Gradienti per lo strato nascosto 1 ---
        # Errore propagato: dL/da1 = delta_2 * W2
        error_h1 = np.dot(delta_2, self.W2)  # (batch_size, neuronsN)

        # Errore pre-attivazione: dL/dz1 = error_h1 * relu'(z1)
        delta_1 = error_h1 * relu_derivative(self.z1)  # (batch_size, neuronsN)

        # Gradiente di W1: dL/dW1 = delta_1^T * input_flat
        input_flat = self.input_batch.reshape(-1, self.input_size_flat)
        dW1 = np.dot(delta_1.T, input_flat) / batch_size  # (neuronsN, input_size_flat)

        # Gradiente di b1: media di delta_1 lungo il batch
        db1 = np.mean(delta_1, axis=0)  # (neuronsN,)

        return dW1, db1, dW2, db2, dW_out, db_out

    def _update_parameters(self, dW1, db1, dW2, db2, dW_out, db_out):
        """Aggiorna pesi e bias usando la discesa del gradiente."""
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

        self.W_out -= self.learning_rate * dW_out
        self.b_out -= self.learning_rate * db_out

    def train(self, input_batch, true_labels, max_epochs=100, batch_size=32, patience=5):
        """
        Addestra il modello su un batch di dati.

        Parametri:
            input_batch: array di forma (n_samples, input_rows, input_cols)
            true_labels: array di forma (n_samples,), etichette vere
            max_epochs: numero massimo di epoche
            batch_size: dimensione del batch
            patience: numero di epoche senza miglioramento prima di fermarsi

        Restituisce:
            list, storia della perdita per epoca
        """
        n_samples = input_batch.shape[0]
        losses = []
        best_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(max_epochs):
            # Mescola i dati
            indices = np.random.permutation(n_samples)
            input_shuffled = input_batch[indices]
            labels_shuffled = true_labels[indices]

            epoch_loss = 0
            for i in range(0, n_samples, batch_size):
                # Estrae il batch corrente
                batch_inputs = input_shuffled[i:i + batch_size]
                batch_labels = labels_shuffled[i:i + batch_size]

                # Passaggio forward
                predictions = self._forward(batch_inputs)

                # Calcola la perdita
                loss = cross_entropy_loss(predictions, batch_labels)
                epoch_loss += loss * batch_inputs.shape[0]

                # Passaggio backward
                dW1, db1, dW2, db2, dW_out, db_out = self._backward(batch_labels)

                # Aggiorna i parametri
                self._update_parameters(dW1, db1, dW2, db2, dW_out, db_out)

            # Calcola la perdita media per l'epoca
            epoch_loss /= n_samples
            losses.append(epoch_loss)

            # Stampa il progresso
            if (epoch + 1) % 10 == 0:
                print(f"Epoca {epoch + 1}/{max_epochs}, Perdita: {epoch_loss:.4f}")

            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping dopo {epoch + 1} epoche")
                    break

        return losses


# --- Funzione per generare dati sintetici ---
def generate_synthetic_data(n_samples, input_shape, n_classes):
    """
    Genera un dataset sintetico semplice.

    Parametri:
        n_samples: numero di campioni
        input_shape: tupla (righe, colonne)
        n_classes: numero di classi

    Restituisce:
        tuple (inputs, labels), input di forma (n_samples, input_rows, input_cols) e
        etichette di forma (n_samples,)
    """
    inputs = np.zeros((n_samples, input_shape[0], input_shape[1]))
    labels = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        label = i % n_classes
        # Genera pattern distinti per ogni classe
        if label == 0:
            inputs[i, :10, :10] = 1.0  # Quadrato in alto a sinistra
        elif label == 1:
            inputs[i, -10:, -10:] = 1.0  # Quadrato in basso a destra
        else:
            inputs[i, 10:20, 10:20] = 1.0  # Quadrato centrale
        labels[i] = label

    return inputs, labels


# --- Esempio di utilizzo ---
if __name__ == '__main__':
    # Parametri
    input_shape = (32, 32)
    num_neurons = 100
    num_outputs = 3  # 3 classi per il dataset sintetico
    lr = 0.001
    batch_size = 32
    max_epochs = 100

    # Crea il modello
    model = Model(input_shape=input_shape, neuronsN=num_neurons, outputsN=num_outputs, learning_rate=lr)

    # Genera dati sintetici
    n_samples = 1000
    X, y = generate_synthetic_data(n_samples, input_shape, num_outputs)

    # --- Prima dell'addestramento ---
    print("--- Prima dell'addestramento ---")
    sample_input = X[:5]  # Primi 5 campioni
    initial_predictions = model.predict(sample_input)
    print(f"Prime 5 predizioni (probabilità):\n{initial_predictions}")
    print(f"Classi predette: {np.argmax(initial_predictions, axis=1)}")
    print(f"Etichette vere: {y[:5]}")

    # --- Addestramento ---
    print("\n--- Addestramento ---")
    losses = model.train(X, y, max_epochs=max_epochs, batch_size=batch_size, patience=10)

    # --- Dopo l'addestramento ---
    print("\n--- Dopo l'addestramento ---")
    final_predictions = model.predict(sample_input)
    print(f"Prime 5 predizioni (probabilità):\n{final_predictions}")
    print(f"Classi predette: {np.argmax(final_predictions, axis=1)}")
    print(f"Etichette vere: {y[:5]}")

    # --- Visualizza la curva di perdita ---
    plt.plot(losses)
    plt.xlabel("Epoca")
    plt.ylabel("Perdita (Cross-Entropy)")
    plt.title("Curva di perdita durante l'addestramento")
    plt.grid(True)
    plt.show()