import numpy as np

class SVM:
    def __init__(self,eta, C, niter, batch_size, verbose):
        self.eta = eta; self.C = C; self.niter = niter; self.batch_size = batch_size; self.verbose = verbose

    def make_one_versus_all_labels(self, y, m):
        """
        y : numpy array of shape (n,)
        m : int (in this homework, m will be 10)
        returns : numpy array of shape (n,m)
        """
        
        # Initialize n x m array full of -1
        a = np.zeros((y.shape[0], m)) - 1
        
        # Iterate through labels in y
        for idx, c in enumerate(y):
            # Mark correct class with a 1
            a[idx,c] = 1
            
        return a

    def compute_loss(self, x, y):
        """
        x : numpy array of shape (minibatch size, 401)
        y : numpy array of shape (minibatch size, 10)
        returns : float
        """

        # Compute regularization term
        left_term = np.linalg.norm(self.w)**2 / 2
        
        # Compute hinge loss term
        right_term = self.C / x.shape[0] * np.sum(np.maximum(0, 1 - np.dot(x, self.w) * y)**2)

        return left_term + right_term
    
    def compute_gradient(self, x, y):
        """
        x : numpy array of shape (minibatch size, 401)
        y : numpy array of shape (minibatch size, 10)
        returns : numpy array of shape (401, 10)
        """
        
        # The gradient of the regularization term is simply the weights
        grad = np.copy(self.w)
        
        # Compute l(w; (x,y))
        # Used when deriving the max function: max(0, a)
        a = (1 - np.dot(x, self.w) * y)        
        
        # Add the gradient of the hinge loss term
        for j in range(self.w.shape[1]):
            for k in range(self.w.shape[0]):
                grad[k, j] += 2 * self.C / x.shape[0] * np.sum(-y[:, j] * x[:, k] * (a[:, j] > 0))
        
        return grad 

    # Batcher function
    def minibatch(self, iterable1, iterable2, size=1):
        l = len(iterable1)
        n = size
        for ndx in range(0, l, n):
            index2 = min(ndx + n, l)
            yield iterable1[ndx: index2], iterable2[ndx: index2]

    def infer(self, x):
        """
        x : numpy array of shape (number of examples to infer, 401)
        returns : numpy array of shape (number of examples to infer, 10)
        """
        
        # Initialize array of predictions, fill with -1
        inference = np.zeros((x.shape[0], self.m)) - 1
        
        # Iterate through samples
        for idx, xi in enumerate(x):
            scores = [0 for i in range(self.m)]
            # Iterate through classes
            for j in range(self.m):
                # Calculate <wj, xi>
                scores[j] = np.dot(xi, self.w[:, j])
            # Predict class with highest score
            inference[idx, np.argmax(scores)] = 1
            
        return inference

    def compute_accuracy(self, y_inferred, y):
        """
        y_inferred : numpy array of shape (number of examples, 10)
        y : numpy array of shape (number of examples, 10)
        returns : float
        """
        correct = 0
        
        # Iterate through rows of y_inferred
        for idx, row in enumerate(y_inferred):
            # Compare to corresponding label y
            # Prediction is correct if the "1" is in the same spot
            if np.array_equal(row, y[idx, :]):
                correct += 1
                
        return correct/y.shape[0]

    def fit(self, x_train, y_train, x_test, y_test):
        """
        x_train : numpy array of shape (number of training examples, 401)
        y_train : numpy array of shape (number of training examples, 10)
        x_test : numpy array of shape (number of training examples, 401)
        y_test : numpy array of shape (number of training examples, 10)
        returns : float, float, float, float
        """
        self.num_features = x_train.shape[1]
        self.m = y_train.max() + 1
        y_train = self.make_one_versus_all_labels(y_train, self.m)
        y_test = self.make_one_versus_all_labels(y_test, self.m)
        self.w = np.zeros([self.num_features, self.m])

        for iteration in range(self.niter):
            # Train one pass through the training set
            for x, y in self.minibatch(x_train, y_train, size=self.batch_size):
                grad = self.compute_gradient(x,y)
                self.w -= self.eta * grad

            # Measure loss and accuracy on training set
            train_loss = self.compute_loss(x_train,y_train)
            y_inferred = self.infer(x_train)
            train_accuracy = self.compute_accuracy(y_inferred, y_train)

            # Measure loss and accuracy on test set
            test_loss = self.compute_loss(x_test,y_test)
            y_inferred = self.infer(x_test)
            test_accuracy = self.compute_accuracy(y_inferred, y_test)

            if self.verbose:
                print("Iteration %d:" % iteration)
                print("Train accuracy: %f" % train_accuracy)
                print("Train loss: %f" % train_loss)
                print("Test accuracy: %f" % test_accuracy)
                print("Test loss: %f" % test_loss)
                print("")
                
        return train_loss, train_accuracy, test_loss, test_accuracy

if __name__ == "__main__":
    # Load the data files
    print("Loading data...")
    x_train = np.load("train_features.npy")
    x_test = np.load("test_features.npy")
    y_train = np.load("train_labels.npy")
    y_test = np.load("test_labels.npy")

    print("Fitting the model...")
    svm = SVM(eta=0.001, C=30, niter=200, batch_size=5000, verbose=True)
    train_loss, train_accuracy, test_loss, test_accuracy = svm.fit(x_train, y_train, x_test, y_test)

    # # to infer after training, do the following:
    # y_inferred = svm.infer(x_test)

    # to compute the gradient or loss before training, do the following:
    y_train_ova = svm.make_one_versus_all_labels(y_train, 10) # one-versus-all labels
    svm.w = np.zeros([401, 10])
    grad = svm.compute_gradient(x_train, y_train_ova)
    loss = svm.compute_loss(x_train, y_train_ova)
