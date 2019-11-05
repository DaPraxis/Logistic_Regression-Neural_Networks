import numpy as np
from l2_distance import l2_distance
import utils as ut
import matplotlib
import matplotlib.pyplot as plt

def classification_rate(valid_pred, valid_labels):
    counter=0
    for i in range(len(valid_labels)):
        if valid_labels[i][0]==valid_pred[i][0]:
            counter+=1
    rate = counter/len(valid_labels)
    return rate

def graph_knn(k_arr, rates, name):
    fig, ax = plt.subplots()
    ax.scatter(k_arr, rates)

    ax.set(xlabel='k', ylabel='Classification Rate',
        title='Classification Rate vs K for '+name+' Datasets')
    ax.grid(False)
    for i, txt in enumerate(rates):
        ax.annotate(txt, (k_arr[i], rates[i]))
    fig.savefig("knn_"+name+".png")
    
    plt.show()

def run_multi_knn(k_arr, train_data, train_labels, valid_data, valid_labels, test_data, test_labels):
    rates = []
    for k in k_arr:
        valid_pred = run_knn(k, train_data, train_labels, valid_data)
        rate = classification_rate(valid_pred, valid_labels)
        rates.append(rate)
    print(rates)
    graph_knn(k_arr, rates, "train-validation")
    rates = []
    for k in k_arr:
        test_pred = run_knn(k, train_data, train_labels, test_data)
        rate = classification_rate(test_pred, test_labels)
        rates.append(rate)
    graph_knn(k_arr, rates, "test")
        


def run_knn(k, train_data, train_labels, valid_data):
    """Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples, 
          and M is the number of features per example.

    Inputs:
        k:            The number of neighbours to use for classification 
                      of a validation example.
        train_data:   The N_TRAIN x M array of training
                      data.
        train_labels: The N_TRAIN x 1 vector of training labels
                      corresponding to the examples in train_data 
                      (must be binary).
        valid_data:   The N_VALID x M array of data to
                      predict classes for.

    Outputs:
        valid_labels: The N_VALID x 1 vector of predicted labels 
                      for the validation data.
    """

    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:,:k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # note this only works for binary labels
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1,1)

    return valid_labels

if __name__ == "__main__":
    test_inputs, test_targets = ut.load_test()
    train_inputs, train_targets = ut.load_train()
    train_inputs_small, train_targets_small = ut.load_train_small()
    valid_inputs, valid_targets = ut.load_valid()
    # test, train, train_small, validate=load_data()
    k_arr = [1,3,5,7,9]
    # run_multi_knn(k_arr, train["train_inputs"], train["train_targets"], validate["valid_inputs"], validate["valid_targets"])
    run_multi_knn(k_arr, train_inputs, train_targets, valid_inputs, valid_targets, test_inputs, test_targets)