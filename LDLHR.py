import pickle
import os
import numpy as np
import multiprocessing 
from ldl_hr import LDL_HR


def save_dict(dataset, scores, name):
    with open(dataset + "//" + name, 'wb') as f:
        pickle.dump(scores, f)

def load_dict(dataset, name):
    file_name = dataset + "//" + name
    if not os.path.exists(file_name):
        file_name += ".pkl"
        
    with open(file_name, 'rb') as f:
        return pickle.load(f)
    
    
def expected_zero_one_loss(Y_pre, Y):
    Y_l = np.argmax(Y_pre, 1)
    return 1 - Y[np.arange(Y.shape[0]), Y_l].mean(0)


def zero_one_loss(Y_pre, Y):
    Y_l_pre = np.argmax(Y_pre, 1)
    Y_l = np.argmax(Y, 1)
    
    return 1 - (Y_l_pre == Y_l).mean()
    

def run_LDLHR(dataset, i, train_x, train_y, test_x, test_y):
    #tune parameters 

    l1 = 0.0001
    l2 = 10000
    l3 = 1
    rho = 0.01
    model = LDL_HR(train_x, train_y, l1, l2, l3, rho)
    model.fit()
    y_pre = model.predict(test_x)

    return y_pre

    

def run_KF(dataset):
    print(dataset)
    X, Y = np.load(dataset + "//feature.npy"), np.load(dataset + "//label.npy")

    train_inds = load_dict(dataset, "train_fold")
    test_inds = load_dict(dataset, "test_fold")


    for i in range(10):
        print("fold", i + 1)

        train_x, train_y = X[train_inds[i + 1]], Y[train_inds[i + 1]]
        test_x, test_y = X[test_inds[i + 1]], Y[test_inds[i + 1]]
        
        y_pre = run_LDLHR(dataset, i, train_x, train_y, test_x, test_y)

        print("0/1 loss:", zero_one_loss(y_pre, test_y))
        print("error probability:", expected_zero_one_loss(y_pre, test_y))      


if __name__ == "__main__":
    datasets = ["SJAFFE"]
    
    for dataset in datasets:
        run_KF(dataset)

