import numpy as np


class RegressionModel:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.weight = np.array([np.random.randn(), np.random.randn()])
        self.previous_loss = 0

    @staticmethod
    def _get_loss(true_prediction_vec, estimated_prediction_vec):
        return np.sum((true_prediction_vec - estimated_prediction_vec) ** 2) / len(true_prediction_vec)

    @staticmethod
    def compute_gradient(true_prediction_vec, estimated_prediction_vec, feature_vec):
        return feature_vec.T.dot(estimated_prediction_vec - true_prediction_vec) / len(true_prediction_vec)

    def update_weights(self, gradient_vec):
        self.weight = self.weight - self.learning_rate * gradient_vec

    def fit(self, feature_vec, true_prediction_vec, thresh):
        epoch_num = 1
        while True:
            prediction_vec = self.predict(feature_vec)
            gradient_vec = self.compute_gradient(true_prediction_vec, prediction_vec, feature_vec)
            self.update_weights(gradient_vec)

            loss = self._get_loss(true_prediction_vec, prediction_vec)
            loss_delta = abs(self.previous_loss - loss)
            print(f"Epoch {epoch_num}: wages - {self.weight}. Loss - {loss}. Loss delta - {loss_delta}.")

            if loss_delta <= thresh:
                break

            self.previous_loss = loss
            epoch_num += 1

    def predict(self, feature_vec):
        return feature_vec.dot(self.weight)

    def test(self, feature_vec, true_prediction_vec):
        predicted_vec = self.predict(feature_vec)
        return self._get_loss(true_prediction_vec, predicted_vec)

    def get_line(self, x):
        return x * self.weight[0] + self.weight[1]

    def get_plot_coords(self, x1, x2):
        return [x1, x2], [self.get_line(x1), self.get_line(x2)]
