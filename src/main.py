from sklearn.preprocessing import LabelEncoder
from protoN import Neuron
import numpy as np
import pandas as pd

# reduit de façon expaonentielle le taux d'apprendisage a partir du nombre d'iterations
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


# Load CSV, encode, split into quarters, slice into time-windowed batches.
# Returns:
#     train_batches: list of DataFrames (training mini-batches)
#     test_batches: list of DataFrames (test mini-batches)
#     input_size: int (total number of columns including label)
    
def prepare_data(csv_path, time_window=50):

    df = pd.read_csv(csv_path)
    cols_dec = verify_type(df)
    df_copy = encoded(df, cols_dec)
    # Drop IP columns — LabelEncoder creates fake ordinal relationships
    df_copy = df_copy.drop(columns=['sourceIP', 'destinationIP'])
    size_input = len(df_copy.columns)

    # Shuffle rows before splitting to avoid temporal distribution bias
    df_copy = df_copy.sample(frac=1, random_state=42).reset_index(drop=True)

    split = int(len(df_copy) * 0.75)

    # Training data: 75% (random batches since data is shuffled)
    df_quart = df_copy.iloc[:split]
    batch_size = max(1, len(df_quart) // 200)  # ~200 batches
    train_batches = [df_quart.iloc[i:i+batch_size] for i in range(0, len(df_quart), batch_size)]

    # Test data: 25%
    df_test = df_copy.iloc[split:]
    test_batches = [df_test.iloc[i:i+batch_size] for i in range(0, len(df_test), batch_size)]

    return train_batches, test_batches, size_input


def run_training(train_batches, test_batches, input_size,
                 num_epochs=3, learning_rate=0.05, decay_rate=0.95,
                 decay_steps=100, threshold_percentile=0.95,
                 on_batch=None, on_epoch=None, on_test_batch=None,
                 on_done=None, should_stop=None):

    # input_size - 2: subtract label column AND dt column
    neuron = Neuron(input_size - 2, learning_rate)
    iterations = 0

    # Compute normalization stats once on full training data (excluding dt and label)
    all_train = np.concatenate([np.array(b.iloc[:, 1:-1]).astype(float) for b in train_batches], axis=0)
    neuron.fit_normalize(all_train)
    del all_train

    for epoch in range(num_epochs):
        if should_stop and should_stop():
            break

        epoch_losses = []
        epoch_accurancy = []

        np.random.shuffle(train_batches)
        for i, df_data in enumerate(train_batches):
            if should_stop and should_stop():
                break

            x_train = np.array(df_data.iloc[:, 1:-1]).astype(float)  # skip dt (col 0)
            y_train = np.array(df_data.iloc[:, -1]).astype(float)

            iterations += 1

            loss = neuron.train_step(x_train, y_train)
            epoch_losses.append(loss)

            x_norm = neuron.normalize(x_train)
            prediction = (neuron.model(x_norm) > 0.5).astype(int).flatten()
            accuracy = np.mean(prediction == y_train)
            epoch_accurancy.append(accuracy)

            if on_batch:
                on_batch(epoch, i, len(train_batches), loss, accuracy)

        avg_loss = np.mean(epoch_losses)
        avg_accu = np.mean(epoch_accurancy)

        if on_epoch:
            on_epoch(epoch, avg_loss, avg_accu)
        else:
            print(f"===Iteration {epoch+1} completed: average loss={avg_loss:.6f}; avgerage accuracy={avg_accu:.4f}===")

    if should_stop and should_stop():
        return

    # Phase de test
    test_losses = []
    test_accuracy = []

    for i, df_batches in enumerate(test_batches):
        if should_stop and should_stop():
            break

        x_test = np.array(df_batches.iloc[:, 1:-1]).astype(float)  # skip dt (col 0)
        y_test = np.array(df_batches.iloc[:, -1]).astype(float)

        loss, prediction = neuron.score_test(x_test, y_test)
        accuracy = np.mean(prediction == y_test)
        test_losses.append(loss)
        test_accuracy.append(accuracy)

        if on_test_batch:
            on_test_batch(i, len(test_batches), loss, accuracy)

    avg_test_loss = np.mean(test_losses) if test_losses else 0
    avg_test_acc = np.mean(test_accuracy) if test_accuracy else 0

    if on_done:
        on_done(avg_test_loss, avg_test_acc)
    else:
        print(f"\n{'='*60}")
        print("TEST RESULTS")
        print(f"{'='*60}")
        print(f"Moyenne de test loss: {(avg_test_loss*100):.1f}%")
        print(f"Moyenne de test accuracy: {(avg_test_acc*100):.1f}%")


if __name__ == '__main__' :
    train_batches, test_batches, size_input = prepare_data("../dataset_sdn.csv")
    print(f"Total des paquets collectées: {len(train_batches)}")
    print(f"Total de paquets de test: {len(test_batches)}")

    run_training(train_batches, test_batches, size_input,
                 num_epochs=100, learning_rate=0.001)