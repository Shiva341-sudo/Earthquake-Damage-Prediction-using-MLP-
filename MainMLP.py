from random import seed, random, randrange 
from pathlib import Path 
import pandas as pd 
from math import exp 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 

def dataset_minmax(dataset): 
  return [[min(col), max(col)] for col in zip(*dataset)] 
 
def dataset_minmax(dataset): 
    minmax = [] 
    for i in range(len(dataset[0]) - 1): 
        col = [row[i] for row in dataset] 
        minmax.append([min(col), max(col)]) 
    return minmax 

def normalize_dataset(dataset, minmax): 
    for row in dataset: 
        for i in range(len(minmax)): 
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0]) 

def cross_validation_split(dataset, n_folds): 
    dataset_split = [] 
    dataset_copy = list(dataset) 
    fold_size = len(dataset) // n_folds 
    for _ in range(n_folds): 
        fold = [] 
        while len(fold) < fold_size: 
            index = randrange(len(dataset_copy)) 
            fold.append(dataset_copy.pop(index)) 
        dataset_split.append(fold) 
    return dataset_split 

def accuracy_metric(actual, predicted): 
    correct = sum(1 for i in range(len(actual)) if actual[i] == predicted[i]) 
    return correct / float(len(actual)) * 100.0 
 
def evaluate_algorithm(dataset, algorithm, n_folds, *args): 
    folds = cross_validation_split(dataset, n_folds) 
    scores = [] 
    for fold in folds: 
        train_set = list(folds) 
        train_set.remove(fold) 
        train_set = sum(train_set, []) 
        test_set = [] 
        for row in fold: 
            row_copy = list(row) 
            test_set.append(row_copy) 
            row_copy[-1] = None 
        predicted = algorithm(train_set, test_set, *args) 
        actual = [row[-1] for row in fold] 
        scores.append(accuracy_metric(actual, predicted)) 
    return scores 

def activate(weights, inputs): 
    activation = weights[-1] 
    for i in range(len(weights) - 1): 
        activation += weights[i] * inputs[i] 
    return activation 

 
def transfer(activation): 
    return 1.0 / (1.0 + exp(-activation)) 

def forward_propagate(network, row): 
    inputs = row 
    for layer in network: 
        new_inputs = [] 
        for neuron in layer: 
            activation = activate(neuron['weights'], inputs) 
            neuron['output'] = transfer(activation) 
            new_inputs.append(neuron['output']) 
        inputs = new_inputs 
    return inputs 

def transfer_derivative(output): 
    return output * (1.0 - output) 
 
def backward_propagate_error(network, expected): 
    for i in reversed(range(len(network))): 
        layer = network[i] 
        errors = [] 
        if i != len(network) - 1: 
            for j in range(len(layer)): 
                error = sum(neuron['weights'][j] * neuron['delta'] 
                            for neuron in network[i + 1]) 
                errors.append(error) 
        else: 
            for j in range(len(layer)): 
                errors.append(layer[j]['output'] - expected[j]) 

        for j in range(len(layer)): 
            neuron = layer[j] 
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output']) 

def update_weights(network, row, l_rate): 
    for i in range(len(network)): 
        inputs = row[:-1] 
        if i != 0: 
            inputs = [neuron['output'] for neuron in network[i - 1]] 
        for neuron in network[i]: 
            for j in range(len(inputs)): 
                neuron['weights'][j] -= l_rate * neuron['delta'] * inputs[j] 
            neuron['weights'][-1] -= l_rate * neuron['delta'] 

 
def train_network(network, train, l_rate, n_epoch, n_outputs): 
    error_history = [] 
    for epoch in range(n_epoch): 
        sum_error = 0 
        for row in train: 
            outputs = forward_propagate(network, row[:-1]) 
            expected = [0 for _ in range(n_outputs)] 
            expected[int(row[-1])] = 1  
            sum_error += sum((expected[i] - outputs[i]) ** 2 for i in range(n_outputs)) 
            backward_propagate_error(network, expected) 
            update_weights(network, row, l_rate) 
        error_history.append(sum_error) 

        if epoch % 100 == 0: 
            print(f'>epoch={epoch}, lrate={l_rate:.3f}, error={sum_error:.3f}') 
    return error_history 

 
def initialize_network(n_inputs, n_hidden, n_outputs): 
    network = [] 
    hidden_layer = [{'weights': [random() for _ in range(n_inputs + 1)]} 
                    for _ in range(n_hidden)] 
    network.append(hidden_layer) 
    output_layer = [{'weights': [random() for _ in range(n_hidden + 1)]} 
                    for _ in range(n_outputs)] 
    network.append(output_layer) 
    return network 


def predict(network, row): 
    outputs = forward_propagate(network, row) 
    return outputs.index(max(outputs)) 

     
def back_propagation(train, test, l_rate, n_epoch, n_hidden): 
    n_inputs = len(train[0]) - 1 
    n_outputs = len(set(int(row[-1]) for row in train))  
    network = initialize_network(n_inputs, n_hidden, n_outputs) 
    history = train_network(network, train, l_rate, n_epoch, n_outputs) 
    predictions = [] 
    for row in test: 
        predictions.append(predict(network, row[:-1])) 
    return predictions, history 

 
def evaluate_algorithm_with_metrics(dataset, algorithm, n_folds, *args): 
    folds = cross_validation_split(dataset, n_folds) 
    scores = [] 
    all_actual = [] 
    all_predicted = [] 
    final_history = [] 


    for fold in folds: 
        train_set = list(folds) 
        train_set.remove(fold) 
        train_set = sum(train_set, []) 
        test_set = [list(row) for row in fold] 
        for row in test_set: row[-1] = None 
        
        predicted, history = algorithm(train_set, test_set, *args) 
        actual = [row[-1] for row in fold] 

        all_actual.extend(actual) 
        all_predicted.extend(predicted) 
        scores.append(accuracy_metric(actual, predicted)) 
        final_history = history  

    return scores, all_actual, all_predicted, final_history 

seed(1) 
path = Path("/nepal_earthquakes_1990_2026.csv") 

dataset = pd.read_csv(path) 

numeric_features = ['latitude', 'longitude', 'depth', 'nst', 'gap', 'dmin', 'rms'] 

 dataset_clean = dataset[numeric_features + ['mag']].dropna() 

dataset_clean['is_strong'] = (dataset_clean['mag'] >= 4.5).astype(int)  

dataset_clean = dataset_clean[numeric_features + ['is_strong']].values.tolist() 

minmax = dataset_minmax(dataset_clean) 

normalize_dataset(dataset_clean, minmax) 

n_folds = 5 
l_rate = 0.1 
n_epoch = 800 
n_hidden = 10 

scores, actuals, preds, history = evaluate_algorithm_with_metrics( 

    dataset_clean, back_propagation, n_folds, l_rate, n_epoch, n_hidden 

) 

print('Scores:', scores) 

print('Mean Accuracy: %.3f%%' % (sum(scores) / len(scores))) 

plt.figure(figsize=(10, 5)) 

plt.plot(history) 

plt.title('Training Error Over Epochs (Final Fold)') 

plt.xlabel('Epoch') 

plt.ylabel('Sum Squared Error') 

plt.grid(True) 

plt.show() 

 
cm = confusion_matrix(actuals, preds) 

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Weak', 'Strong']) 

disp.plot(cmap=plt.cm.Blues) 

plt.title('Confusion Matrix: Earthquake Intensity Prediction') 

plt.show() 
