import numpy as np
import pandas as pd

class Neuron:
    def __init__(self, input, learnig_rate=0.01):
        self.input = input
        self.lr = learnig_rate
        input_size = len(input)
        self.W = np.random.randn(input_size, 1) * 0.1
        self.b = np.zeros(input_size)
        # analyse de le seuil d'anomalie dynamique et changable avec le temp
        self.ret_loss = []
        self.threshold = 0.1
    
    #calcule la fonction d'activation
    def model(self, entry):
        z = entry.dot(self.W) + self.b
        return self.sigmoid(z)
    
    def sigmoid(self, z):
        return 1 / (1+ np.exp(-np.clip(z, -500, 500))) # prevents overflow
    
    # mesure quant bien la prediction des donnes sont arrivées en rapport à le resultat réel
    # plus proches de 0 meilleur la prediction
    def log_loss(self, model, label):
        y = label.reshape(-1 ,1)
        epsilon = 1e-15 
        model = np.clip(model, epsilon, 1-epsilon)
        return 1/ len(y) * np.sum(-y * np.log(model) - (1-y) * np.log(1-model))
    
    # mesure et optimise les futures calcules
    def gradient(self, model, data, label):
        y = label.reshape(-1,1)
        dW = 1/len(y) * np.dot(data.T, model - y)
        db = 1/len(y) - np.sum(model - y)
        return dW, db
    
    def predict(self, model, context=0):
        return (model > (self.threshold + context)).astype(int).ravel()
    
    def update(self, model, data, label):
        dW, db = self.gradient(model, data, label)
        self.W -= self.lr * dW
        self.b -= self.lr * db
        
    def set_threshold(self, percent=0.95):
        if len(self.ret_loss) > 0:
            self.threshold = np.percentile(list(self.ret_loss), percent)
    
    # fonction principal d'evaluation de l'objet;
    # prends comment argument une fonction, la donnée à être evalué, 
    # et applique la prediction aux ensemble des données convertis par la fonction
    def evaluate(self, input, function=None, train=False, label=None):
        data = function(input) if function!=None else input
        model = self.model(data)
        if train:
            loss = self.log_loss(model, label)
            self.ret_loss.append(loss)
            self.update(model=model, data=data, label=label)
            self.set_threshold()
            return loss
        return self.predict(model=model)
        