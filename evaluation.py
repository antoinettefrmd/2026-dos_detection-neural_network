"""
Class Evaluation :

"""
from collections import deque
import numpy as np
import history
import pandas as pd

class NetPerceptron:
    def __init__(self, fields, hidden_size=None, learning_rate=0.01):
        self.lr = learning_rate
        self.fields = fields
        # for (rate, entropy, recency)
        feature_per_field = 3
        input_size = len(fields) * feature_per_field
        self.hidden_size = hidden_size or max(2, input_size //2)
        # encode weight & bias
        self.ew = np.random.randn(input_size, self.hidden_size) * 0.1
        self.eb = np.zeros(self.hidden_size)
        # decode weight & bias
        self.dw = np.random.randn(self.hidden_size, input_size) * 0.1
        self.db = np.zeros(input_size)
        # memory - keeps track of previous data entry
        self.hist = {f: history.History() for f in fields}
        # anomaly threshold
        self.retain_error = deque(maxlen=100000)
        self.threshold = None

    def extract_feat(self, data, t=False):
        feature = []
        for field in self.fields:
            value = data.get(field)
            if value is not None:
                if t: self.hist[field].updat_h(value)
                rate = self.hist[field].avg_rate(value)
                
                # entropy and diversity for measrue
                etrp = self.hist[field].get_entropy()
                
                # measure how recently was this value seen
                recy = self.hist[field].get_recency(value)
                
                # DEBUG: Check for None values
                if rate is None:
                    rate = 0.0
                #    print(f"WARNING: rate is None for field={field}, value={value}")
                if etrp is None:
                    etrp = 0.0
                #    print(f"WARNING: entropy is None for field={field}")
                if recy is None:
                    recy = 0.0
                #    print(f"WARNING: recency is None for field={field}, value={value}")

                feature.append(rate)
                feature.append(etrp)
                feature.append(recy)
                
            else:
                # for rate, entropy, recency
                feature.extend([0.0, 0.0, 0.0])
                
        # DEBUG: Final check
        #print(f"Feature list before array conversion: {feature}")
        
        feature_array = np.array(feature, dtype=np.float64)
        # DEBUG: Check for problems in the array
        
        if np.any(np.isnan(feature_array)):
            print(f"WARNING: NaN detected in feature array!")
    
        #print(f"Final feature array: {feature_array}")
        return feature_array
    
    def set_threshold(self, percentile=95):
        # threshold is defined based in the errors retains in an normal traffic
        if len(self.retain_error) > 0:
            self.threshold = np.percentile(list(self.retain_error), percentile)
        else:
            self.threshold = 0.1 # default value
    
    def forward(self, x):
        # encode
        coded = self.sigmoid(np.dot(x, self.ew) + self.eb)
        # decode
        decoded = self.sigmoid(np.dot(coded, self.dw) + self.db)
        
        return coded, decoded
        
    def sigmoid(self, z):
        return 1/(1 + np.exp(-np.clip(z, -500, 500))) # prevent overflow
    
    def training_steps(self, message):
        x = self.extract_feat(message, True)
        
        coded, decoded = self.forward(x)
        
        error = x - decoded
        ret_loss = np.mean(error ** 2)
        
        # backpropagation
        output = -2 * error * decoded * (1 - decoded)
        back_dW = np.outer(coded, output)
        back_dB = output
        
        # hidden layers gradient
        l_hidden = np.dot(output, self.dw.T) * coded * (1-coded)
        back_eW = np.outer(x, l_hidden)
        back_eB = l_hidden
        
        self.ew -= self.lr * back_eW
        self.eb -= self.lr * back_eB
        self.dw -= self.lr * back_dW
        self.db -= self.lr * back_dB
        
        self.retain_error.append(ret_loss)
        
        return ret_loss
    
    def predict(self, message):
        x = self.extract_feat(message)
        _ , decoded = self.forward(x)
        
        score = np.mean((x - decoded) ** 2)
        
        if self.threshold is None:
            self.set_threshold()
        is_anomaly = score > self.threshold
        return (is_anomaly, score)
    
    
if __name__ == "__main__":
    fields=["stime", "proto_number", "saddr", "daddr", "dport", "state_number", "dur", "spkts", "sbytes"]
    detector = NetPerceptron(fields=fields,
        hidden_size=3, learning_rate=0.05)
    
    print(f"Number of fields: {len(detector.fields)}")
    print(f"Input size: {detector.ew.shape[0]}")  # Should be 27 (9 fields × 3)
    print(f"Hidden size: {detector.hidden_size}")
    print(f"ew shape: {detector.ew.shape}")  # Should be (27, hidden_size)
    print(f"dw shape: {detector.dw.shape}")  # Should be (hidden_size, 27)
    
    df = pd.read_csv('databaseDoS.csv')
    
    # Lire le CSV
    print(f"Données totales: {len(df)} lignes")
    print(f"Colonnes: {df.columns.tolist()}")
    
    # Séparer en 4 quarts
    quart = len(df) // 4
    i = 0
    checkpoint = [100, 1000, 1500, 2000, 5000]
    # 1er quart pour l'entraînement
    df_train = df.iloc[:quart]
    print("fedding the machine ...")
    for _, data in df_train.iterrows():
        i += 1
        message_input = {field : data[field] for field in fields}
        loss = detector.training_steps(message_input)
        if i in checkpoint :
            print(f'loss state: {loss:.6f}')
    detector.set_threshold()
    
    # 2ème quart pour le test
    print("--------test area----------")
    df_test = df.iloc[quart:2*quart]
    anomaly = 0
    not_anomaly = 0
    for _, data in df_test.iterrows():
        test_input = {field : data[field] for field in fields}
        is_anomaly, score = detector.predict(test_input)
        if is_anomaly: anomaly += 1 
        else: not_anomaly +=1
    print(f'anomaly: {anomaly}; not anomaly: {not_anomaly}')
        