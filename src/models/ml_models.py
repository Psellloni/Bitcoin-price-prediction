from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

class RandomForest:
    def __init__(self, **kwargs):
        self.model = RandomForestRegressor(**kwargs)

    def train(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        self.model.fit(X_train, y_train)

    def eval(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

        y_pred = self.model.predict(X_test)

        return classification_report(y_test, y_pred)
    
    def plot_learning_curves(self):
        pass


