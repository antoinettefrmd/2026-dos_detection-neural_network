import numpy as np
class Neuron:
    def __init__(self, input_size, learnig_rate=0.01):
        self.lr = learnig_rate
        self.W = np.random.randn(input_size, 1) * 0.1
        self.b = 0
        self.mean = None
        self.std = None
        self.ret_loss = []
        self.anomaly_threshold = None

        # Initialisation ADAM
        self.t = 0 # nb d'itération
        self.beta1 = 0.9 # Taux de décroissance pour l'estimation du premier moment (moyenne mobile)
        self.beta2 = 0.999 # # Taux de décroissance pour l'estimation du second moment (variance mobile)
        self.adam_eps = 1e-8
        self.mW = np.zeros_like(self.W)  # Accumule la direction des gradients (fisrt moment)
        self.vW = np.zeros_like(self.W) # Mesure la dispersion des gradients (Second moment)
        self.mb = 0 # Premier moment (moyenne) des gradients pour le biais b
        self.vb = 0 # Second moment (variance non centrée) des gradients pour le biais b
        
    def sigmoid(self, z):
        return 1 / (1+ np.exp(-np.clip(z, -500, 500))) # prevents overflow
    
    #calcule la fonction d'activation dans le reseau
    def model(self, entry):
        z = entry.dot(self.W)
        z += self.b
        return self.sigmoid(z)
    
    # mesure quant bien la prediction des donnes sont arrivées en rapport à le resultat réel
    # plus proches de 0 meilleur la prediction
    def log_loss(self, model, label):
        y = label.reshape(-1 ,1)
        s = len(y)
        epsilon = 1e-15 
        model = np.clip(model, epsilon, 1-epsilon)
        return 1/s * np.sum(-y * np.log(model) - (1-y) * np.log(1-model))
    
    # mesure et optimise les futures calcules
    def gradient(self, model, data, label):
        y = label.reshape(-1,1)
        s = len(y)
        dW = 1/s * np.dot(data.T, model - y)
        db = 1/s * np.sum(model - y)
        return dW, db
    
    def update(self, dW, db):
        max_grad = 5.0
        # Limite sur les gradient pour eviter l'explosion des gradients
        dW = np.clip(dW, -max_grad, max_grad)
        db = np.clip(db, -max_grad, max_grad)
        
        self.t += 1
        
        # Récupération de la direction moyenne en intégrant le gradient courant  
        self.mW = self.beta1 * self.mW + (1 - self.beta1) * dW
        self.mb = self.beta1 * self.mb + (1 - self.beta1) * db
        
        # Récupération de la dispersion moyenne en intégrant le gradient courant 
        self.vW = self.beta2 * self.vW + (1 - self.beta2) * (dW ** 2)
        self.vb = self.beta2 * self.vb + (1 - self.beta2) * (db ** 2)
        
        # Réajustement, car au début m et v sont initialisé à 0
        # Correction en donnant plus de poids aux premiers gradients puis diminution
        mW_hat = self.mW / (1 - self.beta1 ** self.t)
        mb_hat = self.mb / (1 - self.beta1 ** self.t)
        vW_hat = self.vW / (1 - self.beta2 ** self.t)
        vb_hat = self.vb / (1 - self.beta2 ** self.t)
        
        # pas adaptatif = (direction moyenne) / (racine de la variance)
        self.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.adam_eps)
        self.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.adam_eps)

    # Normalise toutes les données avant le trainnig pour qu'elle ait plus de sens   
    def fit_normalize(self, X_all):
        self.mean = np.mean(X_all, axis=0)
        self.std = np.std(X_all, axis=0)
    
    # Normalise les données en urgence si ce n'est pas déjà fait !
    def normalize(self, input_train):
        if self.mean is None or self.std is None:
            self.mean = np.mean(input_train, axis=0)
            self.std = np.std(input_train, axis=0)
        return (input_train - self.mean) / (self.std + 1e-8)
    
    # fonction principal d'entrainement du perceptron;
    # prends comment argument une fonction, la donnée à être evalué, 
    # et applique la prediction aux ensemble des données convertis par la fonction
    def train_step(self, input, label, function=None):
        data = function(input) if function != None else input
        
        data = self.normalize(data)
        
        model = self.model(data)
        
        # calcule la divergence du clacule
        loss = self.log_loss(model=model, label=label)
        self.ret_loss.append(loss)
        
        # propagation et mise à jour de poids et bias
        dW, db = self.gradient(model=model,data=data,label=label)
        self.update(dW, db)
        
        return loss
    
    # Cet fonction evalue une donnée passée comment entrée converti par le parametre function
    # retourne si le donnée est une anomalie et quant distant il est de la normalité
   
    def score_test(self, input, label, function=None):
        data = function(input) if function != None else input
        data = self.normalize(data)
        model = self.model(data)
        
        loss = self.log_loss(model=model, label=label)
        prediction = (model > 0.5).astype(int).flatten()
        return loss, prediction
