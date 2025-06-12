import math
import numpy as np

class NN:
    def __init__(self, layers):
        self.layers = []
        self.acts = []
        self.layers_len = layers

        self.layers.append(np.array([]))
        self.acts.append(self.sigmoid)
        for i in range(len(layers[1])):
            self.layers.append(np.array([]))
            self.acts.append(self.sigmoid)
        self.acts[-1] = self.softmax


    # установка весов
    def set_weights(self, weights):
        weight_ind = 0
        if len(self.layers_len[1]) == 0:
            self.layers[0] = np.array(weights).reshape(self.layers_len[2], self.layers_len[0])
        else:
            self.layers[0] = np.array(weights[:self.layers_len[0] * self.layers_len[1][0]]).reshape(self.layers_len[1][0], self.layers_len[0])
            weight_ind += self.layers_len[0] * self.layers_len[1][0]
            for i in range(len(self.layers_len[1])):
                if i == len(self.layers_len[1]) - 1:
                    self.layers[i + 1] = np.array(weights[weight_ind:weight_ind + self.layers_len[1][i] * self.layers_len[2]]).reshape(self.layers_len[2], self.layers_len[1][i])
                else:
                    self.layers[i + 1] = np.array(weights[weight_ind:weight_ind + self.layers_len[1][i] * self.layers_len[1][i + 1]]).reshape(self.layers_len[1][i + 1], self.layers_len[1][i])
                    weight_ind += self.layers_len[1][i] * self.layers_len[1][i + 1]


    # получение общего количества весов
    def get_total_weights(self):
        if len(self.layers_len[1]) == 0:
            return self.layers_len[0] * self.layers_len[2]
        else:
            weights = self.layers_len[0] * self.layers_len[1][0]
            for i in range(len(self.layers_len[1])):
                if i == len(self.layers_len[1]) - 1:
                    weights += self.layers_len[1][i] * self.layers_len[2]
                else:
                    weights += self.layers_len[1][i] * self.layers_len[1][i + 1]

            return weights


    # функции активации
    def sigmoid(self, x):
        return [1 / (1 + math.exp(-x_i)) for x_i in x]


    def relu(self, x):
        x[x < 0] = 0
        return x


    def softmax(self, x):
        e_x = np.exp(x)
        pred = [e_i / np.sum(e_x) for e_i in e_x]
        # return np.where(pred == np.max(pred))[0][0], pred
        return pred


    # возвращает действие для агента
    def predict(self, inputs):
        f = inputs.astype('float64')
        for i in range(len(self.layers)):
            f = self.acts[i](self.layers[i] @ f)

        return f