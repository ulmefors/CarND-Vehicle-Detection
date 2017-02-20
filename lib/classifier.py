from sklearn.svm import LinearSVC


class Classifier:

    def __init__(self, clf=None):
        self.clf = clf

    def train(self, X_train, X_test, y_train, y_test, type='SVC'):
        # Use a linear SVC (support vector classifier)
        svc = LinearSVC()

        print('Start training')
        # Train the SVC
        svc.fit(X_train, y_train)
        self.clf = svc

        # Evaluate on test set
        accuracy_test = svc.score(X_test, y_test)
        print('Test accuracy of SVC: {0:.2f}%'.format(accuracy_test * 100))

        return accuracy_test

    def predict(self, features):
        return self.clf.predict(features)

    def get_classifier(self):
        return self.clf


def main():
    pass


if __name__ == "__main__":
    main()
