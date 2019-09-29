import numpy as np
import math 
import timeit
import sys

np.set_printoptions(threshold=sys.maxsize)

######## DO NOT MODIFY THIS FUNCTION ########
def draw_rand_label(x, label_list):
    seed = abs(np.sum(x))
    while seed < 1:
        seed = 10 * seed
    seed = int(1000000 * seed)
    np.random.seed(seed)
    return np.random.choice(label_list)
#############################################


class Q1:

    def feature_means(self, iris):
        return np.mean(iris[:,:4], axis = 0)

    def covariance_matrix(self, iris):
        return np.cov(iris[:,:4].T)

    def feature_means_class_1(self, iris):
        feature_class_1 = iris[iris[:,-1]==1]
        return np.mean(feature_class_1[:,0:4], axis = 0)

    def covariance_matrix_class_1(self, iris):
        feature_class_1 = iris[iris[:,-1]==1]
        return np.cov(feature_class_1[:,:4].T)


class HardParzen:
    def __init__(self, h):
        self.h = h

    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.label_list = len(np.unique(train_labels))

    def distance(self, x , Y, p=2):
        return (np.sum((np.abs(x - Y)) ** p, axis=1)) ** (1.0 / p)

    def compute_predictions(self, test_data):
        num_test = test_data.shape[0]
        counts = np.zeros((num_test, self.label_list))
        classes_pred = np.zeros(num_test)

        for (i, ex) in enumerate(test_data):
            #i is the row index
            #ex is the i'th row
            distances = self.distance(ex, self.train_inputs)
            index_neighbors = []
            radius = self.h
            for j in range(len(distances)):
                #for j from 0 to 150
                if distances[j] < radius:
                    index_neighbors.append(j)
            close_neighbors = list(self.train_labels[index_neighbors]-1)
            
            for j in range(len(close_neighbors)):
                counts[i, int(close_neighbors[j])] += 1
            if len(index_neighbors) == 0:
                classes_pred[i] = draw_rand_label(ex, self.label_list)
            else:
                classes_pred[i] = np.argmax(counts[i, :]) +1
        return classes_pred


class SoftRBFParzen:
    def __init__(self, sigma):
        self.sigma  = sigma

    def train(self, train_inputs, train_labels):
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        self.label_dim = train_inputs.shape[1]
        self.label_list = len(np.unique(train_labels))

    def distance(self, x , Y, p=2):
        return (np.sum((np.abs(x - Y)) ** p, axis=1)) ** (1.0 / p)

    def compute_predictions(self, test_data):
        weights = []
        num_test = test_data.shape[0]
        pred_label = np.zeros(num_test)

        for (i, ex) in enumerate(test_data):
            distances = self.distance(ex, self.train_inputs)
            prediction = np.zeros(self.label_list)
            for (index, distance) in enumerate(distances):
                weight = self.calculate_weight(distance)
                label = self.train_labels[index]
                prediction[int(label -1)] += weight
            pred_label[i] = np.argmax(prediction)+1
        return pred_label
        
    def calculate_weight(self, distance):
        first_coefficient =  1 / ((2*math.pi)**(distance/2))*self.sigma
        second_coefficient = math.e**(-(1/2)*(distance**2 / self.sigma**2))
        return first_coefficient * second_coefficient


def split_dataset(iris):
    training = np.empty((0,5))
    validation = np.empty((0,5))
    test = np.empty((0,5))

    for (index,line) in enumerate(iris):
        if (index % 5 == 0) or (index % 5 == 1) or (index % 5 == 2):
            training = np.append(training, [line], axis=0)
        if (index % 5 == 3):
            validation = np.append(validation, [line], axis=0)
        if (index % 5 == 4):
            test = np.append(test, [line], axis=0)
    whole = [training, validation, test]
    split = tuple(whole)
    return split


class ErrorRate:
    def __init__(self, x_train, y_train, x_val, y_val):
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def hard_parzen(self, h):
        self.h = h 
        f = HardParzen(h)
        f.train(self.x_train, self.y_train)
        predicted_labels = f.compute_predictions(self.x_val)
        number_of_predictions = len(self.x_val)
        good_predictions = 0
        for i in range(len(predicted_labels)):
            if predicted_labels[i] == self.y_val[i]:
                good_predictions += 1.0
        return 1.0 - (good_predictions/number_of_predictions)


    def soft_parzen(self, sigma):
        self.sigma = sigma
        e = SoftRBFParzen(sigma)
        e.train(self.x_train, self.y_train)
        predicted_labels = e.compute_predictions(self.x_val)
        number_of_predictions = len(predicted_labels)
        good_predictions = 0

        for i in range(len(predicted_labels)):
            if predicted_labels[i] == self.y_val[i]:
                good_predictions += 1.0
        #print(good_predictions)
        #print(number_of_predictions)
        return (1.0 - (good_predictions/number_of_predictions))*2


def get_test_errors(iris):
    radius_hard = [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 15.0, 20.0]
    sigma_soft = [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 15.0, 20.0]
    validation_error_hard = []
    validation_error_soft = []
    splitted_data = split_dataset(iris)
    x_train = splitted_data[0][:, 0:4]
    y_train = splitted_data[0][:, 4]
    x_val = splitted_data[1][:, 0:4]
    y_val = splitted_data[1][:, 4]
    x_test = splitted_data[2][:, 0:4]
    y_test = splitted_data[2][:, 4]
    good_predictions_hard = 0
    good_predictions_soft = 0
    number_of_predictions = len(x_test)
    a = ErrorRate(x_train, y_train, x_val, y_val)

    for radius in radius_hard:
        validation_error_hard.append(a.hard_parzen(radius))
    selected_radius = radius_hard[np.argmin(validation_error_hard)]

    for sigma in sigma_soft:
        validation_error_soft.append(a.soft_parzen(sigma))
    selected_sigma = sigma_soft[np.argmin(validation_error_soft)]

    b = HardParzen(selected_radius)
    b.train(x_train, y_train)
    pred_labels_hard = b.compute_predictions(x_test)

    for i in range(len(pred_labels_hard)):
        if pred_labels_hard[i] == y_test[i]:
            good_predictions_hard += 1.0
    class_error_hard = 1.0 - (good_predictions_hard/number_of_predictions)

    c = SoftRBFParzen(selected_sigma)
    c.train(x_train, y_train)
    pred_labels_soft = c.compute_predictions(x_test)

    for i in range(len(pred_labels_soft)):
        if pred_labels_soft[i] == y_test[i]:
            good_predictions_soft += 1.0
    class_error_soft = 1.0 - (good_predictions_soft/number_of_predictions)
    errors = np.array([class_error_hard, class_error_soft])
    return errors 


def random_projections(X, A):
    new_matrix = np.zeros((X.shape[0], A.shape[1]))
    new_matrix = np.dot(X, A) / np.sqrt(2)
    return new_matrix


iris = np.loadtxt('iris.txt')
#f = HardParzen(1)
#f.train(iris[5:,:4] ,iris[5:,4])

#print(f.compute_predictions(iris[0:5,:4]))
#split_dataset(iris)
#e = SoftRBFParzen(2)
#e.train(iris[:,:4] ,iris[:,4])
#print(e.compute_predictions(iris[:,:4]))
#g = ErrorRate(iris[:,:4], iris[:,4], iris[:,:4], iris[:,4])
#print(g.soft_parzen(1))
#get_test_errors(iris)

#Question 9
N = 500
splitted_data = split_dataset(iris)
training = splitted_data[0]
validation = splitted_data[1]
test = splitted_data[2]
x_train = splitted_data[0][:, 0:4]
y_train = splitted_data[0][:, 4]
x_val = splitted_data[1][:, 0:4]
y_val = splitted_data[1][:, 4]
validation_error = np.zeros((500, 9))
validation_error_soft = np.zeros((500, 9))
hyperparameter = [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 15.0, 20.0]


for i in range(N):
    random_matrix = np.random.normal(0, 1, (4, 2))
    modified_training = random_projections(x_train, random_matrix)
    modified_validation = random_projections(x_val, random_matrix)
    a = ErrorRate(modified_training, y_train, modified_validation, y_val)
    for (index, value) in enumerate(hyperparameter):
        validation_error[i, index] = a.hard_parzen(value)

for i in range(N):
    random_matrix = np.random.normal(0, 1, (4, 2))
    modified_training = random_projections(x_train, random_matrix)
    modified_validation = random_projections(x_val, random_matrix)
    a = ErrorRate(modified_training, y_train, modified_validation, y_val)
    for (index, value) in enumerate(hyperparameter):
        validation_error_soft[i, index] = a.soft_parzen(value)    



  