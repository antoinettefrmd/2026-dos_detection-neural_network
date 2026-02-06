import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# ------------------------------------------------------------------
# 1. CHARGER ET PRÉPARER LES DONNÉES
# ------------------------------------------------------------------

# Lire le CSV
df = pd.read_csv('databaseDoS.csv')

print(f"Données totales: {len(df)} lignes")
print(f"Nombre de colonnes: {len(df.columns)}")
print(f"Colonnes: {df.columns.tolist()}")

# ------------------------------------------------------------------
# 2. PRÉPARATION DES DONNÉES AVEC ENCODAGE DES IP
# ------------------------------------------------------------------

# Identifier les colonnes de type object (chaînes de caractères)
object_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nColonnes textuelles à encoder: {object_cols}")

# Encoder chaque colonne textuelle
encoders = {}
for col in object_cols:
    print(f"Encodage de la colonne '{col}'...")
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))  # Convertir en string puis encoder
    encoders[col] = le  # Sauvegarder l'encodeur pour plus tard
    
    # Afficher quelques informations
    unique_vals = len(df[col].unique())
    print(f"  {unique_vals} valeurs uniques encodées")

# ------------------------------------------------------------------
# 3. SÉPARATION DES DONNÉES
# ------------------------------------------------------------------

# Séparer en 4 quarts
quart = len(df) // 4

# 1er quart pour l'entraînement
df_train = df.iloc[:quart]
# 2ème quart pour le test
df_test = df.iloc[quart:2*quart]

print(f"\n1er quart (train): {len(df_train)} lignes")
print(f"2ème quart (test): {len(df_test)} lignes")

# Préparer X_train, y_train (TOUTES les colonnes sauf la dernière)
X_train = df_train.iloc[:, :-1].values  # Toutes sauf label
y_train = df_train.iloc[:, -1].values   # label

# Préparer X_test, y_test
X_test = df_test.iloc[:, :-1].values
y_test = df_test.iloc[:, -1].values

print(f"\nShape X_train: {X_train.shape}")  # Devrait être (262143, nb_colonnes-1)
print(f"Shape y_train: {y_train.shape}")

# ------------------------------------------------------------------
# 4. NORMALISATION (MAINTENANT TOUT EST NUMÉRIQUE)
# ------------------------------------------------------------------

print("\nNormalisation des données...")

# Calculer la moyenne et l'écart-type sur le train
X_train_mean = np.mean(X_train, axis=0)
X_train_std = np.std(X_train, axis=0)

# Vérifier qu'il n'y a pas de colonnes avec écart-type nul
zero_std_cols = np.where(X_train_std == 0)[0]
if len(zero_std_cols) > 0:
    print(f"Attention: {len(zero_std_cols)} colonnes avec écart-type nul")
    print(f"Indices: {zero_std_cols}")
    # Ajouter une petite valeur pour éviter la division par zéro
    X_train_std[zero_std_cols] = 1.0

# Normaliser
X_train = (X_train - X_train_mean) / (X_train_std + 1e-8)
X_test = (X_test - X_train_mean) / (X_train_std + 1e-8)

print("Normalisation terminée avec succès!")

# ------------------------------------------------------------------
# 5. FONCTIONS DU NEURONE (inchangées)
# ------------------------------------------------------------------

def initialisation(X):
    W = np.random.randn(X.shape[1], 1)
    b = np.random.randn(1)
    return W, b

def model(X, W, b):
    Z = X.dot(W) + b
    A = 1 / (1 + np.exp(-Z))
    return A

def log_loss(A, y):
    y = y.reshape(-1, 1)
    epsilon = 1e-15
    A = np.clip(A, epsilon, 1 - epsilon)
    return 1 / len(y) * np.sum(-y * np.log(A) - (1 - y) * np.log(1 - A))

def gradients(A, X, y):
    y = y.reshape(-1, 1)
    dW = 1 / len(y) * np.dot(X.T, A - y)
    db = 1 / len(y) * np.sum(A - y)
    return dW, db

def update(dW, db, W, b, learning_rate):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b

def predict(X, W, b):
    A = model(X, W, b)
    return (A >= 0.5).astype(int).ravel()

# ------------------------------------------------------------------
# 6. ENTRAÎNEMENT SUR LE 1ER QUART
# ------------------------------------------------------------------

print("\n" + "="*50)
print("ENTRAÎNEMENT SUR LE 1ER QUART")
print("="*50)

W, b = initialisation(X_train)
losses = []

for i in range(15000):
    A = model(X_train, W, b)
    loss = log_loss(A, y_train)
    losses.append(loss)
    
    dW, db = gradients(A, X_train, y_train)
    W, b = update(dW, db, W, b, learning_rate=0.05)
    
    if i % 20 == 0:
        print(f"Itération {i}: loss = {loss:.4f}")

# ------------------------------------------------------------------
# 7. TEST SUR LE 2ÈME QUART
# ------------------------------------------------------------------

print("\n" + "="*50)
print("TEST SUR LE 2ÈME QUART")
print("="*50)

# Prédictions sur le test set
y_pred_test = predict(X_test, W, b)
accuracy_test = accuracy_score(y_test, y_pred_test)

print(f"Accuracy sur le test set: {accuracy_test:.4f}")
print(f"Précision: {accuracy_test:.2%}")

# Comparaison avec le train
y_pred_train = predict(X_train, W, b)
accuracy_train = accuracy_score(y_train, y_pred_train)
print(f"Accuracy sur le train set: {accuracy_train:.4f}")

# ------------------------------------------------------------------
# 8. VISUALISATIONS
# ------------------------------------------------------------------

# Graphique 1: Évolution de la loss
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(losses)
plt.title('Évolution de la Loss pendant l\'entraînement')
plt.xlabel('Itérations')
plt.ylabel('Loss')
plt.grid(True, alpha=0.3)

# Graphique 2: Points en 2D (2 premières features - maintenant encodées)
plt.subplot(1, 3, 2)

# Afficher les vrais labels
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', 
            alpha=0.6, s=10, label='Vrais labels')
plt.title('Données TEST - Vrais labels\n(Adresses IP encodées)')
plt.xlabel('IP Source (encodée)')
plt.ylabel('IP Destination (encodée)')
plt.colorbar(label='Classe')
plt.legend()

# Graphique 3: Points avec prédictions
plt.subplot(1, 3, 3)

# Afficher les prédictions
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_test, cmap='coolwarm',
            alpha=0.6, s=10, label='Prédictions')
plt.title('Données TEST - Prédictions\n(Adresses IP encodées)')
plt.xlabel('IP Source (encodée)')
plt.ylabel('IP Destination (encodée)')
plt.colorbar(label='Prédiction')
plt.legend()

plt.tight_layout()
plt.show()