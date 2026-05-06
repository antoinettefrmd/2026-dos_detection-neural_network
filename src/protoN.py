import numpy as np
class Neuron:

    # Pour choisir la taille des différentes couches on va tester différentes config pour l'instant
    # Trop de neuronnes entraineraot un sur apprentissage 
    # Mais pas assez entraînerait un sous apprentissage 
    def __init__(self, input_size, learnig_rate=0.001, hidden1_size=32 ,hidden2_size=16, l2_lambda=0.001):
        self.lr = learnig_rate
        # On a décidé de rajouter une couche pour que notre modèle distingues des patterns complexes pas que linéaires
        # On ne multiplie plus par 0.1 mais par un nombre qui s'adapte à la taille de la couches
        # Meilleure équilibre des poids
        self.l2_lambda = l2_lambda

        # Taille de la couche en fonction des inputs, initialisation des poids et du biais 
        self.W1 = np.random.randn(input_size, hidden1_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden1_size))
        # Couche cachée 1
        self.W2 = np.random.randn(hidden1_size, hidden2_size) * np.sqrt(2.0 / hidden1_size)
        self.b2 = np.zeros((1, hidden2_size))
        # Couche cachée 2
        self.W3 = np.random.randn(hidden2_size, 1) * np.sqrt(2.0 / hidden2_size)
        self.b3 = np.zeros((1, 1))

        self.mean = None
        self.std = None
        self.ret_loss = []

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
        for name in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
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
    # la première couche utilise ReLU pour introduire de la non-linéarité,
    # La dernière couche applique une sigmoid afin de produire une probabilité en sortie.
    def model(self, entry):
        self.z1 = entry.dot(self.W1) + self.b1
        self.a1 = self.relu(self.z1)

        # La sortie de la première couche devient l'entrée de la deuxième

        self.z2 = self.a1.dot(self.W2) + self.b2
        self.a2 = self.relu(self.z2)

        self.z3 = self.a2.dot(self.W3) + self.b3

        return self.sigmoid(self.z3)
    
    # mesure quant bien la prediction des donnes sont arrivées en rapport à le resultat réel
    # plus proches de 0 meilleur la prediction
    def log_loss(self, model, label):
        y = label.reshape(-1 ,1)
        s = len(y)
        epsilon = 1e-15 
        model = np.clip(model, epsilon, 1-epsilon)
        return 1/s * np.sum(-y * np.log(model) - (1-y) * np.log(1-model))
    
    # mesure et optimise les futures calculs
    def gradient(self, model, data, label):
        y = label.reshape(-1,1)
        s = len(y)

        # calcul erreur prédiction - vérité
        # dérivation de loss + signmoid
        # Le gradient est la dérivée de la loss
        dZ3 = model - y

        dW3 = 1/s * np.dot(self.a2.T, dZ3) + self.l2_lambda * self.W3
        db3 = 1/s * np.sum(dZ3, axis=0, keepdims=True)

        dA2 = dZ3.dot(self.W3.T)
        # Correction des neuronnes qui ont servi seulement, les autres morts = pas de correction
        dZ2 = dA2 * self.relu_derivative(self.z2)
        dW2 = 1/s * np.dot(self.a1.T, dZ2) + self.l2_lambda * self.W2
        db2 = 1/s * np.sum(dZ2, axis=0, keepdims=True)
        
        dA1 = dZ2.dot(self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.z1)
        dW1 = 1/s * np.dot(data.T, dZ1) + self.l2_lambda * self.W1
        db1 = 1/s * np.sum(dZ1, axis=0, keepdims=True)
        
        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}

    
    def update(self, grads):
        self.t += 1
        for name in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
            # Limite sur les gradients pour eviter l'explosion des gradients
            g = np.clip(grads[name], -5.0, 5.0)
            # Récupération de la direction moyenne en intégrant le gradient courant
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g
            # Récupération de la dispersion moyenne en intégrant le gradient courant
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (g ** 2)
            
            # Réajustement, car au début m et v sont initialisé à 0
            # Correction en donnant plus de poids aux premiers gradients puis diminution
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            param = getattr(self, name)
            # pas adaptatif = (direction moyenne) / (racine de la variance)
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
