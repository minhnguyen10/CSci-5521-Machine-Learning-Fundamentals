import numpy as np
np.random.seed(123)

def process_data(data,mean=None,std=None):
    # normalize the data to have zero mean and unit variance (add 1e-15 to std to avoid numerical issue)
    if mean is not None:
        # directly use the mean and std precomputed from the training data
        data = (data - mean) / std
        return data
    else:
        # compute the mean and std based on the training data
        mean = np.mean(data, axis= 0)
        std =  np.std(data, axis=0)+ 1e-15
        data = (data - mean) / std  
        return data, mean, std

def process_label(label):
    # convert the labels into one-hot vector for training
    one_hot = np.zeros([len(label),10])
    for i in range(len(label)):
        one_hot[i][label[i]] = 1
    return one_hot

def tanh(x):
    # implement the hyperbolic tangent activation function for hidden layer
    # You may receive some warning messages from Numpy. No worries, they should not affect your final results
    # f_x = x # placeholder
    f_x = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

    return f_x

def softmax(x): #nx10
    # implement the softmax activation function for output layer
    # f_x = x # placeholder
    e_x = np.exp(x)
    f_x = np.exp(x)/e_x.sum(axis=1, keepdims=True)

    return f_x

class MLP:
    def __init__(self,num_hid):
        # initialize the weights
        self.weight_1 = np.random.random([64,num_hid])
        self.bias_1 = np.random.random([1,num_hid])
        self.weight_2 = np.random.random([num_hid,10])
        self.bias_2 = np.random.random([1,10])
        self.num_hid = num_hid

    def fit(self,train_x,train_y, valid_x, valid_y):
        # learning rate
        lr = 5e-3
        # counter for recording the number of epochs without improvement
        count = 0
        best_valid_acc = 0

        """
        Stop the training if there is no improvment over the best validation accuracy for more than 50 iterations
        """
        while count<=50:
            # training with all samples (full-batch gradient descents)
            # implement the forward pass (from inputs to predictions)
            # call predict() to get the redicted y for each iter
            zh = self.get_hidden(train_x)
            foward_y = self.predict(train_x)
            y = process_label(foward_y)
            Eyder = train_y - y
            zder = 1 - zh**2 

            
            # delta_v = lr * np.dot(zh.T, Eyder) #num_hidx10
            delta_w2 = lr * zh.T.dot(Eyder)
            delta_w2_bias = lr * np.sum(Eyder, axis=0)

            rmy_v = Eyder.dot(self.weight_2.T)
            prod = rmy_v*zder
            delta_w1 = lr * prod.T.dot(train_x)
            
            delta_w1_bias = lr* np.sum(prod, axis=0) 


            #update the parameters based on sum of gradients for all training samples
            self.weight_2 = self.weight_2 + delta_w2  #update weight 2
            self.weight_1 = self.weight_1 + delta_w1.T #update weight 1
            self.bias_2 = self.bias_2 + delta_w2_bias  #update bias 2
            self.bias_1 = self.bias_1 + delta_w1_bias #update bias 1

            # evaluate on validation data
            predictions = self.predict(valid_x)
            valid_acc = np.count_nonzero(predictions.reshape(-1)==valid_y.reshape(-1))/len(valid_x)

            # compare the current validation accuracy with the best one
            if valid_acc>best_valid_acc:
                best_valid_acc = valid_acc
                count = 0
            else:
                count += 1

        return best_valid_acc

    def predict(self,x):
        # generate the predicted probability of different classes
        # call get_hidden() to compute z
        
        zh = self.get_hidden(x)
        
        ot = zh.dot(self.weight_2) + self.bias_2
        y = softmax(ot)
        y_prob = np.argmax(y, axis=1)

        # return y
        return y_prob

    def get_hidden(self,x):
        # extract the intermediate features computed at the hidden layers (after applying activation function)
        # compute z 
        z = x.dot(self.weight_1) + self.bias_1
        z = tanh(z)
        return z


    def params(self):
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2
