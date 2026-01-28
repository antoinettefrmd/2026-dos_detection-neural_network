"""
-> data clustering - define a arithimetic average of a data type.
-> n-clustering neighbor: after n interactions of clustering the algorithm should be able to define 
an interval in which every data that falls outside is considered abnormal.
"""
import numpy as np
import history

class NetPerceptron:
    def __init__(self, fields, learning_rate=0.01):
        self.lr = learning_rate
        self.fields = fields
        self.w = np.zeros(len(fields))   # input weight
        self.b = 0.0                    # bias
        self.hist = {field: history.History() for field in fields}                 # memory - keeps track of previous data entry

    def extract_feat(self, data):
        feature = []
        for field in self.fields:
            value = data.get(field)
            if value is not None:
                self.hist[field].upda_h(value)
                rate = self.hist[field].avg_rate(value)
            else:
                rate = 0.0
            feature.append(rate)
        return np.array(feature)
        
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def predict(self, x):
        z = np.dot(self.w, x) + self.b
        return self.sigmoid(z)
    
    def training(self, message, target):
        x = self.extract_feat(message)
        output = self.predict(x)
        error = target - output
        
        self.w += self.lr * error * x
        self.b += self.lr * error