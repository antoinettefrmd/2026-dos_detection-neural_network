import numpy as np
class Neuron:

    # Pour choisir la taille des différentes couches on va tester différentes config pour l'instant
    # Trop de neuronnes entraineraot un sur apprentissage 
    # Mais pas asser entraînerait un sous apprentissage 
    def __init__(self, input_size, learnig_rate=0.001, hidden_size=32 , l2_lambda=0.001):
        self.lr = learnig_rate
        # On a décidé de rajouter deux couches pour que notre modèle distingues des patterns complexes pas que linéaires
        # On ne multiplie plus par 0.1 mais par un nombre qui s'adapte à la taille de la couches
        # Meilleure équilibre des poids
        self.l2_lambda = l2_lambda
        # Taille de la couche en fonction des inputs, initialisation des poids et du biais 
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        # Couche cachée 1 
        self.W2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, 1))

        self.mean = None
        self.std = None
        self.ret_loss = []
        self.anomaly_threshold = None

        # Initialisation ADAM
        self.t = 0 # nb d'itération
        self.beta1 = 0.9 # Taux de décroissance pour l'estimation du premier moment (moyenne mobile)
        self.beta2 = 0.999 # # Taux de décroissance pour l'estimation du second moment (variance mobile)
        self.adam_eps = 1e-8
        self._init_adam()
    
    # Initialisation des paramètre ADAM direction et vitesse 
    def _init_adam(self):
        self.m = {}
        self.v = {}
        for name in ['W1', 'b1', 'W2', 'b2']:
            param = getattr(self, name)
            # Pour les poids : 
            # Accumule la direction des gradients (fisrt moment)
            # Mesure la dispersion des gradients (Second moment)

            # Pour les biais
            # Premier moment (moyenne) des gradients pour le biais b
            # Second moment (variance non centrée) des gradients pour le biais b
            self.m[name] = np.zeros_like(param) 
            self.v[name] = np.zeros_like(param) 


    def sigmoid(self, z):
        return 1 / (1+ np.exp(-np.clip(z, -500, 500))) # prevents overflow
    
    # Introduction de la non linéarité 
    # Facilite la descente de gradient
    # Évite les problèmes de saturation
    def relu(self, z):
        return np.maximum(0, z)

    # Permet la rétropropagation (apprentissage du neuronne)
    def relu_derivative(self, z):
        return (z > 0).astype(float)
    

    # Forward pass du réseau de neurones :
    # applique successivement deux transformations linéaires
    # la premieière couche utilisent ReLU pour introduire de la non-linéarité,
    # La dernière couche applique une sigmoid afin de produire une probabilité en sortie.
    def model(self, entry):
        self.z1 = entry.dot(self.W1) + self.b1
        self.a1 = self.relu(self.z1)

        # La sortie de la première couche devient l'entrée de la deuxième

        self.z2 = self.a1.dot(self.W2) + self.b2

        return self.sigmoid(self.z2)
    
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
        # Output layer gradients
        dZ2 = model - y
        dW2 = 1/s * np.dot(self.a1.T, dZ2) + self.l2_lambda * self.W2
        db2 = 1/s * np.sum(dZ2, axis=0, keepdims=True)
        # Hidden layer gradients
        dA1 = dZ2.dot(self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.z1)
        dW1 = 1/s * np.dot(data.T, dZ1) + self.l2_lambda * self.W1
        db1 = 1/s * np.sum(dZ1, axis=0, keepdims=True)
        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
    
    def update(self, grads):
        # max_grad = 5.0
        # # Limite sur les gradients pour eviter l'explosion des gradients
        # dW = np.clip(dW, -max_grad, max_grad)
        # db = np.clip(db, -max_grad, max_grad)
        
        # self.t += 1
        
        # # Récupération de la direction moyenne en intégrant le gradient courant  
        # self.mW = self.beta1 * self.mW + (1 - self.beta1) * dW
        # self.mb = self.beta1 * self.mb + (1 - self.beta1) * db
        
        # # Récupération de la dispersion moyenne en intégrant le gradient courant 
        # self.vW = self.beta2 * self.vW + (1 - self.beta2) * (dW ** 2)
        # self.vb = self.beta2 * self.vb + (1 - self.beta2) * (db ** 2)
        
        # # Réajustement, car au début m et v sont initialisé à 0
        # # Correction en donnant plus de poids aux premiers gradients puis diminution
        # mW_hat = self.mW / (1 - self.beta1 ** self.t)
        # mb_hat = self.mb / (1 - self.beta1 ** self.t)
        # vW_hat = self.vW / (1 - self.beta2 ** self.t)
        # vb_hat = self.vb / (1 - self.beta2 ** self.t)
        
        # # pas adaptatif = (direction moyenne) / (racine de la variance)
        # self.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.adam_eps)
        # self.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.adam_eps)
        self.t += 1
        for name in ['W1', 'b1', 'W2', 'b2']:
            g = np.clip(grads[name], -5.0, 5.0)
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            param = getattr(self, name)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.adam_eps)
            setattr(self, name, param)

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
        grads = self.gradient(model=model,data=data,label=label)
        self.update(grads)
        
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
