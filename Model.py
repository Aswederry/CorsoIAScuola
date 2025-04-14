import numpy as np


class Model:
    def __init__(self, input_shape=(32, 32), neuronsN=100, outputsN=10, learning_rate=0.01):
        self.input_rows, self.input_cols = input_shape
        self.neuronsN = neuronsN
        self.outputsN = outputsN
        self.learning_rate = learning_rate

        self.inputMatrix = np.zeros(input_shape)

        self.hiddenLayer1_weights = np.random.rand(neuronsN, self.input_rows, self.input_cols)

        self.hiddenLayer2_weights = np.random.rand(self.neuronsN, self.neuronsN)

        self.outputLayer_weights = np.random.rand(outputsN, self.neuronsN)

        self.h1_output = None
        self.h2_output = None
        self.raw_output = None
        self.final_output_prob = None

    def FeedInputMatrix(self, inputMatrix):
        inputMatrixNP = np.array(inputMatrix)
        if inputMatrixNP.shape != (self.input_rows, self.input_cols):
            raise ValueError(
                f"Input matrix shape mismatch. Expected {(self.input_rows, self.input_cols)}, got {inputMatrix.shape}")
        self.inputMatrix = inputMatrix

    def Propagation(self):
        # Prima si fa il prodotto tra la matrice di input e la prima matrice nascosta e la somma si mette in h1_output
        self.h1_output = np.sum(self.inputMatrix * self.hiddenLayer1_weights, axis=(1, 2))

        # Secondo si fa il prodotto scalare tra i pesi della seconda matrice nascosta e la somma che abbiamo calcolato qua sopra
        self.h2_output = np.dot(self.hiddenLayer2_weights, self.h1_output)

    def CalculateOutput(self):
        # Questo fa il prodotto scalare tra la matrice di output e il prodotto scalare calcolato sopra
        self.raw_output = np.dot(self.outputLayer_weights, self.h2_output)

        # Questo output sarebbe la somma del prodotto scalare qua sopra
        total_output = np.sum(self.raw_output)

        if total_output == 0:
            self.final_output_prob = np.ones(self.outputsN) / self.outputsN
        else:
            self.final_output_prob = self.raw_output / total_output

        return self.final_output_prob * 100

    def Predict(self, inputMatrix):
        self.FeedInputMatrix(inputMatrix)
        self.Propagation()
        output_percentages = self.CalculateOutput()
        return output_percentages

    def Training(self, index):
        if self.final_output_prob is None:
            raise RuntimeError(
                "Bisogna prima chiamare Predict() prima di questa funzione per avere le probabilità di output")

        gradient = -np.log(index)

        self.hiddenLayer1_weights -= gradient * self.learning_rate
