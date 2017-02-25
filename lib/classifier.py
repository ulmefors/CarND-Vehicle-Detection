from sklearn.svm import LinearSVC


class Classifier:

    def __init__(self, clf=None):
        self.clf = clf

    def train(self, X_train, X_test, y_train, y_test, type='SVC'):
        # Use a linear SVC (support vector classifier)
        self.clf = LinearSVC()

        print('Start training')
        self.clf.fit(X_train, y_train)

        # Evaluate on test set
        accuracy_test = self.clf.score(X_test, y_test)
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
