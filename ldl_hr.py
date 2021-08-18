import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
eps = 1e-15

def Top_K(Y, k):
    Y_k = np.zeros_like(Y)
    k_ind = np.argsort(Y)[:, -k:]
    all_rows = np.arange(Y.shape[0])
    Y_k[all_rows.reshape((-1, 1)), k_ind] = 1 
    
    Y_rest = np.copy(Y)
    Y_rest[np.asarray(Y_k, dtype=bool)] = 0
    return Y_k, Y_rest


def L2_Regularizer(weights):
    norm = (weights ** 2).sum()
    return 0.5 * norm, weights


def KL(y, y_hat):
    y = np.clip(y, eps, 1)
    y_hat = np.clip(y_hat, eps, 1)
    
    loss = -1 * np.sum(y * np.log(y_hat))
    grad = y_hat - y
    
    return loss, grad


def L1(y, y_hat, J = None):
    if J is None:
        J = 1
    
    diff = (y_hat - y) * J
    sign = np.sign(diff)
    
    loss = (np.absolute(diff)).sum()
    
    sign_y_hat = y_hat * sign
    grad = np.sum(-sign_y_hat, 1).reshape((-1, 1)) * y_hat + sign_y_hat
    return loss, grad



def margin(Y, Y_pre, neg_Y, rho):
    g = 0 
    loss = 0
    
    for i in range(Y_pre.shape[1]):
        y = Y[:, i].reshape((-1, 1))
        y_pre= Y_pre[:, i].reshape((-1, 1))
        Y_diff = (y_pre - Y_pre)
        loss_ind = np.asarray(Y_diff < rho, dtype = np.int) * y * neg_Y
        
        loss_ind[:, i] = -1 * loss_ind.sum(1)      
        loss += (-1 * loss_ind[:, i])
        g += loss_ind
    
    Y_pre_g = g * Y_pre
    grad = np.sum(-Y_pre_g, 1).reshape((-1, 1)) * Y_pre + Y_pre_g
    loss = loss.sum() * rho + Y_pre_g.sum()
    
    return loss, grad


def append_intercept(X):
        return np.hstack((X, np.ones(np.size(X, 0)).reshape(-1, 1)))


class LDL_HR:
    def __init__(self, X, Y, l1 = 0, l2 = 0, l3 = 0, rho = 0.01):
        
        self.X = append_intercept(X)
        self.Y = Y
        
        #balancing coefficient
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

        #for maximum margin 
        self.rho = rho     
        self.Y_top1, self.Y_rest = Top_K(self.Y, 1)
        self.neg_Y = 1 - self.Y_top1
        
        #for verborse
        self.iters = 0
        self.l = 0
                
        self.n_examples, self.n_features = self.X.shape
        self.n_outputs = self.Y.shape[1]
        
    
    def output(self, weights):        
        weights = weights.reshape((self.n_features, self.n_outputs))
        Y_hat = softmax(np.dot(self.X, weights), axis = 1)  
        
        #function loss: learn the top1
        fun_loss, fun_grad = L1(self.Y_top1, Y_hat)    
        
        #regularization
        if self.l1 == 0:
            l2_reg, l2_reg_grad = 0, 0
        else:
            l2_reg, l2_reg_grad = L2_Regularizer(weights)

        #large margin
        if self.l2 == 0:
            margin_loss, margin_grad = 0, 0
        else:
            margin_loss, margin_grad = margin(self.Y_top1, Y_hat, self.neg_Y, self.rho)
        
        #learn the rest except the optimal one
        if self.l3 == 0:
            rest_loss, rest_grad = 0, 0
        else:
            rest_loss, rest_grad = L1(self.Y_rest, Y_hat)
        
    
        loss = fun_loss + self.l1 * l2_reg + self.l2 * margin_loss + self.l3 * rest_loss
        grad = self.l1 * l2_reg_grad + np.dot(self.X.T, (fun_grad + self.l2 * margin_grad + self.l3 * rest_grad))
        
        self.l = loss
        
        return loss, grad.reshape((-1, ))
        
    
    def fit(self):
        weights = np.zeros(self.n_features * self.n_outputs) 
        optimize_result = minimize(self.output, weights, method='l-BFGS-b', jac = True, 
                                   #callback = self.func,
                                   options = {'gtol':1e-6, 'disp': False, 'maxiter':1000 })
    
        self.weights = optimize_result.x.reshape((self.n_features, self.n_outputs))       
    
    def func(self, weights):
        self.iters +=1 
        print(self.iters, self.l)
        
        
    def predict(self, test_X):
        test_X = append_intercept(test_X)
        test_Y_hat = softmax(np.dot(test_X, self.weights), axis = 1)  
        return test_Y_hat
        
    def __str__(self):
        model_str = "LDLHR_"
        model_str += str(self.l1) + "_"
        model_str += str(self.l2) + "_"
        model_str += str(self.l3) + "_"
        model_str += str(self.rho) + "_"
        
        return model_str