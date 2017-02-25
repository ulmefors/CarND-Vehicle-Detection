import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class PreProcessor:

    def __init__(self, scaler=None):
        self.X_scaler = scaler

    def scale_features(self, features):
        scaled_features = self.X_scaler.transform(np.array(features).reshape(1, -1))
        return scaled_features

    def preprocess(self, car_features, non_car_features):

        # Create an array stack of feature vectors
        X = np.vstack((car_features, non_car_features)).astype(np.float64)
        # Fit a per-column scaler
        self.X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = self.X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(non_car_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

        return X_train, X_test, y_train, y_test

    def get_scaler(self):
        return self.X_scaler

def main():
    pass


if __name__ == "__main__":
    main()
