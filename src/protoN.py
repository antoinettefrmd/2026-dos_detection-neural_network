from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

class Neuron:
    def __init__(self, input_size, learnig_rate=0.01):
        self.lr = learnig_rate
        self.W = np.random.randn(input_size, 1) * 0.1
        self.b = 0
        self.ret_loss = []
        self.anomaly_threshold = None
        
    def sigmoid(self, z):
        return 1 / (1+ np.exp(-np.clip(z, -500, 500))) # prevents overflow
    
    #calcule la fonction d'activation dans le reseau
    def model(self, entry):
        z = entry.dot(self.W) + self.b
        return self.sigmoid(z)

    
    # mesure quant bien la prediction des donnes sont arrivées en rapport à le resultat réel
    # plus proches de 0 meilleur la prediction
    def log_loss(self, model, label):
        y = label.reshape(-1 ,1)
        s = len(y)
        epsilon = 1e-15 
        model = np.clip(model, epsilon, 1-epsilon)
        return  1/s * np.sum(-y * np.log(model) - (1-y) * np.log(1-model))
    
    # mesure et optimise les futures calcules
    def gradient(self, model, data, label):
        y = label.reshape(-1,1)
        s = len(y)
        dW = 1/s * np.dot(data.T, model - y)
        db = 1/s * np.sum(model - y)
        return dW, db
    
    def update(self, dW, db):
        self.W -= self.lr * dW
        self.b -= self.lr * db
    
    # Met à jour le threshold en fonction de la fct perte
    def set_threshold(self, percent=0.95):
        if len(self.ret_loss) > 0:
            self.anomaly_threshold = np.percentile(list(self.ret_loss), percent * 100)
    
    # retourne si le depasse le seuil
    def predict_anomaly(self, loss):
        if self.anomaly_threshold is None:
            self.set_threshold()
        return loss > self.anomaly_threshold
    
    # fonction principal d'entrainement du perceptron;
    # prends comment argument une fonction, la donnée à être evalué, 
    # et applique la prediction aux ensemble des données convertis par la fonction
    def train_step(self, input, label, function=None):
        data = function(input) if function != None else input
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
        model = self.model(data)
        
        loss = self.log_loss(model=model, label=label)
        prediction = self.predict_anomaly(loss)
        return np.mean(prediction == label.astype(bool))

def slice_per_time(data, time) :
    mask = ((data['stime'] >= time) & (data['stime'] <= time + 500))
    df_ret = data[mask].copy()
    return df_ret, time + 500

def encoded(df):
    for col in df:
        if col == 'saddr' or col == 'daddr':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df

if __name__ == '__main__' :
    df = pd.read_csv("../databaseDoS.csv")
    # Récupération de la bdd dans df
    df = encoded(df)
    
    # Sélection du premier quart pour l'entrainement
    quart = len(df) // 4
    df_quart = df.iloc[:quart]
    
    # Création du neuron
    neuron = Neuron(13, 0.5)

    # Premier temps de la base de donnée pour savoir où on commence
    first_time = df_quart["stime"][1]
    
    df_train, first_time = slice_per_time(df_quart, first_time) # récupération du flux par tranche de 3 minutes

    i = 0
    rest = len(df_quart)
    while len(df_train) > 0 :
        x_train = np.array(df_train.iloc[:,:-1]) # Sélection des flags approprié
        y_train = np.array(df_train.iloc[:,-1]) # colonne des tag 
        for j in range(50):
            log = neuron.train_step(x_train,y_train) #entrainement par tranche
            if (j%10 == 0): print(f"loss precision in {i},{j}: {log:.4f}")
            neuron.lr /= 1.5 # ajustement du learning rate
            df_train, first_time = slice_per_time(df_quart, first_time) # récupération du flux par tranche de 3 minutes

        i+=1
