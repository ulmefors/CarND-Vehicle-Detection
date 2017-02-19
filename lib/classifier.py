from sklearn.svm import LinearSVC


class Classifier:

    # Classifier
    def __init__(self):
        self.clf = None

    def train(self, X_train, X_test, y_train, y_test, type='SVC'):
        # Use a linear SVC (support vector classifier)
        svc = LinearSVC()
        # Train the SVC
        print('Started training')
        svc.fit(X_train, y_train)
        print('Test Accuracy of SVC: {0:.2f}%'.format(svc.score(X_test, y_test) * 100))
        self.clf = svc

    def predict(self, features):
        return self.clf.predict(features)


def main():
    pass


if __name__ == "__main__":
    main()