from sklearn.preprocessing import LabelEncoder
from protoN import Neuron
import numpy as np
import pandas as pd


# reduit de façon exponentielle le taux d'apprendisage a partir du numéro d'iterations
def decaying_lr(learning_rate, iteration, decay_rate=0.95, decay_steps=100):
    return learning_rate * (decay_rate ** (iteration // decay_steps))


def slice_per_time(df, time, time_window=500, min_sockets=1) :
    mask = ((df['dt'] >= time) & (df['dt'] <= time + time_window))
    df_ret = df[mask].copy()
    
    # on agument la fênetre de temps autant que que notre liste des données soient vide
    while len(df_ret) < min_sockets and time < df['dt'].max():
        time += time_window
        mask = ((df['dt'] >= time) & (df['dt'] <= time + time_window))
        df_ret = df[mask].copy()
    
    return df_ret, (time + time_window)


def encoded(df, cols_dec):
    df = df.copy()
    df = df.fillna(0)
    if len(cols_dec) > 0:
        for col in df:
            if col in cols_dec:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
    return df
 
    
def verify_type(df):
    object_type = []
    for col in df.columns:
        # Get pandas dtype for the column
        pandas_dtype = df[col].dtype

        # Determine status
        if pandas_dtype == 'object':
            object_type.append(col)
    return object_type


if __name__ == '__main__' :
    df = pd.read_csv("../dataset_sdn.csv")
    # Récupération de la bdd dans df
    cols_dec = verify_type(df)
    df_copy = encoded(df, cols_dec)
    size_input = len(df.columns)
    # Sélection du premier quart pour l'entrainement
    quart = len(df_copy) // 4
    df_quart = df_copy.iloc[:quart]
    
    # Création du neuron
    iterations = 0
    learning_rate=0.05
    neuron = Neuron(size_input-1, learning_rate)   
    
    # Premier temps de la base de donnée pour savoir où on commence
    first_time = df_quart["dt"].iloc[0]
    all_data = []

    while True :
        df_train, first_time = slice_per_time(df_quart, first_time, time_window=50) # récupération du flux par tranche de 3 minutes
        if len(df_train) == 0: break
        all_data.append(df_train)
    
    print(f"Total des paquets collectées: {len(all_data)}")
    
    num_epoch = 10
    for epoch in range(num_epoch):
        epoch_losses = []
        epoch_accurancy = []
            
        # randomize les paquets par iterations (creer de imprevisibilité)
        np.random.shuffle(all_data)
        for i, df_data in enumerate(all_data):
            x_train = np.array(df_data.iloc[:,:-1]) # Sélection des flags approprié
            y_train = np.array(df_data.iloc[:,-1]) # colonne des tag

            # mettre à jour le taux d'apprendissage
            neuron.lr = decaying_lr(learning_rate, iterations)
            iterations+=1
            
            # entraine une fois par paquets
            loss = neuron.train_step(x_train, y_train)
            epoch_losses.append(loss)
            prediction = (neuron.model(x_train) > 0.5).astype(int).flatten()
            accuracy = np.mean(prediction == y_train)
            epoch_accurancy.append(accuracy)
            if epoch%10==0:
                print(f"Iteration {epoch+1} de {num_epoch}, Paquet {i} de {len(all_data)}:\nloss: {loss:.6f}, accuracy: {accuracy:.4f}")
        
        neuron.set_threshold()
        avg_loss=np.mean(epoch_losses)
        avg_accu=np.mean(epoch_accurancy)
        print(f"===Iteration {epoch+1} completed: average loss={avg_loss:.6f}; , avgerage accuracy={avg_accu:.4f}===")
        
    # ------------------ phase de test ---------------------------------------------
    print("\n" + "="*60)
    print("Phase de Test")
    print("="*60)
    
    df_test = df_copy.iloc[quart:2*quart]
    
    first_time = df_test["dt"].iloc[0]
    test_batches = []
    t=0
    while True :
        df_batch_test, first_time = slice_per_time(df_test, first_time, time_window=50) # récupération du flux par tranche de 3 minutes
        if len(df_batch_test) == 0: break
        test_batches.append(df_batch_test)
        if t%1500==0:
            print(f"size of test_batches at {t}: {len(df_batch_test)}")
        t+=1
    print(f"total de paquets sur df_test: {len(df_test)}") 
    print(f"total de paquets de test: {len(test_batches)}")
    
    test_accuracy = []
    test_losses = []
    test_predictions = []
    test_labels = []
    
    i=0
    for df_batches in test_batches:
        x_test = np.array(df_batches.iloc[:, :-1])
        y_test = np.array(df_batches.iloc[:, -1])
        
        loss , prediction = neuron.score_test(x_test, y_test)
        accuracy = np.mean(prediction == y_test)
        test_accuracy.append(accuracy)
        test_losses.append(loss)
        test_predictions.append(prediction)
        test_labels.extend(y_test)
        pred =  [p for p in prediction if p==True]
        if i%10==0:
            print(f"Test {i} de {len(test_batches)} Prediction is {len(pred)} VS. valeur réel est {(y_test.mean()*100):.1f}%:\n"
                  f"loss: {loss:.6f}; accuracy: {accuracy:.4f}")
            print(f"last label: {y_test[-1]}")
        i+=1
    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    print(f"Moyenne de test loss: {(np.mean(test_losses)*100):.1f}%")
    print(f"Moyenne de test accuracy: {(np.mean(test_accuracy)*100):.1f}%")