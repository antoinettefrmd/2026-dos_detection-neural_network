from sklearn.preprocessing import LabelEncoder
from protoN import Neuron
import numpy as np
import pandas as pd
# reduit de façon exponentielle le taux d'apprendisage a partir du nombre d'iterations
def decaying_lr(learning_rate, iteration, decay_rate=0.95, decay_steps=100):
    return learning_rate * (decay_rate ** (iteration // decay_steps))

def slice_per_time(df, time, time_window=500, min_sockets=1) :
    mask = ((df['dt'] >= time) & (df['dt'] <= time + time_window))
    df_ret = df[mask].copy()
    
    # augmentation de la fênetre de temps autant que que notre liste des données soient vide
    while len(df_ret) < min_sockets and time < df['dt'].max():
        time += time_window
        mask = ((df['dt'] >= time) & (df['dt'] <= time + time_window))
        df_ret = df[mask].copy()
    
    return df_ret, (time + time_window)

def encoded(df):
    df = df.copy()
    for col in df:
        if col == 'sourceIP' or col == 'destinationIP' or col == 'Protocol':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df

def eval_data(df):
    # CHECK LABEL DISTRIBUTION
    print("="*60)
    print("DATA DISTRIBUTION CHECK")
    print("="*60)
    
    # Assuming last column is the label (0=normal, 1=attack)
    label_counts = df.iloc[:, -1].value_counts()
    print(f"\nLabel distribution:")
    print(label_counts)
    print(f"\nPercentages:")
    print(label_counts / len(df) * 100)
    
    total = len(df)
    class_0 = (df.iloc[:, -1] == 0).sum()
    class_1 = (df.iloc[:, -1] == 1).sum()
    
    print(f"\nClass 0 (normal): {class_0} ({class_0/total*100:.2f}%)")
    print(f"Class 1 (attack): {class_1} ({class_1/total*100:.2f}%)")
    
    imbalance_ratio = max(class_0, class_1) / min(class_0, class_1)
    print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 10:
        print("⚠️  WARNING: Severe class imbalance detected!")
        print("   Your model may learn to always predict the majority class.")
    
    print("="*60)

#  
def check_for_leakage(df_train, df_test):
    """
    Check for potential data leakage
    """
    print("\n" + "="*60)
    print("DATA LEAKAGE CHECK")
    print("="*60)
    
    # Check 1: Temporal overlap
    train_time_range = (df_train['dt'].min(), df_train['dt'].max())
    test_time_range = (df_test['dt'].min(), df_test['dt'].max())
    
    print(f"\nTrain time range: {train_time_range}")
    print(f"Test time range: {test_time_range}")
    
    if train_time_range[1] >= test_time_range[0]:
        print("⚠️  WARNING: Temporal overlap detected!")
    else:
        print("✓ No temporal overlap")
    
    # Check 2: Feature overlap (e.g., same IPs)
    train_sourceIPs = set(df_train['sourceIP'].unique())
    test_sourceIPs = set(df_test['sourceIP'].unique())
    
    overlap_ratio = len(train_sourceIPs & test_sourceIPs) / len(test_sourceIPs)
    print(f"\nSource IP overlap: {overlap_ratio*100:.2f}%")
    
    if overlap_ratio > 0.9:
        print("⚠️  WARNING: High IP overlap - model may memorize specific IPs")
    
    # Check 3: Label distribution similarity
    train_attack_rate = df_train.iloc[:, -1].mean()
    test_attack_rate = df_test.iloc[:, -1].mean()
    
    print(f"\nTrain attack rate: {train_attack_rate*100:.2f}%")
    print(f"Test attack rate: {test_attack_rate*100:.2f}%")
    
    if abs(train_attack_rate - test_attack_rate) > 0.2:
        print("⚠️  WARNING: Very different label distributions")
    
    print("="*60)

if __name__ == '__main__' :
    # Récupération de la bdd dans df
    df = pd.read_csv("../dataset_sdn.csv")

    for col in df.columns:
        print(f"{col:15s} | type: {str(df[col].dtype):10s} | exemples: {df[col].iloc[0] if len(df) > 0 else 'vide'}")
    df = encoded(df)
    df.drop(labels=['sourceIP'], axis=1)
    for col in df.columns:
        print(f"{col:15s} | type: {str(df[col].dtype):10s} | exemples: {df[col].iloc[0] if len(df) > 0 else 'vide'}")
    
    # eval_data(df)
    # Sélection du premier quart pour l'entrainement
    quart = len(df) // 4
    df_quart = df.iloc[:quart]
    df_test = df.iloc[quart:2*quart]
    # check_for_leakage(df_quart, df_test)

    # Premier temps de la base de donnée pour savoir où on commence
    first_time = df_quart["dt"].iloc[0]
    all_data = []

    while True :
        df_train, first_time = slice_per_time(df_quart, first_time, time_window=50) # récupération du flux par tranche de 3 minutes
        if len(df_train) == 0: break
        all_data.append(df_train)
    
    print(f"Total des paquets collectées: {len(all_data)}")
    
    # Création du neuron
    iterations = 0
    learning_rate=0.01
    neuron = Neuron(22, learning_rate)    

        
    num_epoch = 100
    for epoch in range(num_epoch):
        epoch_losses = []
        epoch_accurancy = []
        epoch_labels = [] 
            
        # randomize les paquets par iterations (creer de imprevisibilité)
        np.random.shuffle(all_data)
        for i, df_data in enumerate(all_data):
            x_train = np.array(df_data.iloc[:,:-1]) # Sélection des flags approprié
            y_train = np.array(df_data.iloc[:,-1]) # colonne des tag
            # mettre à jour le taux d'apprendissage
            neuron.lr = decaying_lr(learning_rate, iterations)
            iterations+=1
        
            # Force la conversion en nombres
            x_train = x_train.astype(float)  
            y_train = y_train.astype(float)  
            # entraine une fois par paquets
            loss = neuron.train_step(x_train, y_train)
            epoch_losses.append(loss)
            
            prediction = (neuron.model(x_train) > 0.5).astype(int).flatten()
            accuracy = np.mean(prediction == y_train)
            # epoch_accurancy.append(accuracy)
            # if epoch%10==0:
                # print(f"Iteration {epoch+1} de {num_epoch}, Paquet {i} de {len(all_data)}:\nloss: {loss:.6f}, accuracy: {accuracy:.4f}")
        
        avg_loss=np.mean(epoch_losses)
        avg_accu=np.mean(epoch_accurancy)
        print(f"===Iteration {epoch+1} completed: average loss={avg_loss:.6f}; , average accuracy={avg_accu:.4f}===")

    # phase de test
    print("\n" + "="*60)
    print("Phase de Test")
    print("="*60)
    
    first_time = df_test["dt"].iloc[0]
    test_batches = []
    
    while True :
        df_batch, first_time = slice_per_time(df_test, first_time, time_window=500) # récupération du flux par tranche de 3 minutes
        if len(df_batch) == 0: break
        test_batches.append(df_batch)
        
    print(f"test batch: {len(test_batches)}")
    i = 0
    test_accuracy = []
    test_accuracy2 = []
    test_losses = []
    test_predictions = []
    test_labels = []
    for df_batches in test_batches:
        x_test = np.array(df_batches.iloc[:, :-1])
        y_test = np.array(df_batches.iloc[:, -1])
        
        loss , prediction = neuron.score_test(x_test, y_test)
        print(loss)
        accuracy1 = np.mean(prediction == y_test.groupeby )

        test_accuracy.append(accuracy1)
        test_losses.append(loss)
        test_predictions.append(prediction)
        test_labels.extend(y_test)

        if i%50==0:
            print(f"Test {i} de {len(test_batches)} Prediction is {prediction} / valeur réel est {y_test.mean()}:\n"
                f"loss: {loss:.6f} ; accuracy1: {accuracy1:.4f}")
        i +=1
        
    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    print(f"Moyenne de test loss: {np.mean(test_losses):.6f}")
    print(f"Moyenne de test accuracy: {np.mean(test_accuracy):.4f}")
    