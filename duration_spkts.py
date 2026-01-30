import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def analyse_et_visualisation(fichier_csv):
    """
    Charge le CSV, calcule les stats sur colonne11/colonne12 selon le tag,
    puis affiche une visualisation avec les moyennes.
    """

    # ================== CHARGEMENT ==================
    df = pd.read_csv(fichier_csv)

    if df.shape[1] < 16:
        print(f"Erreur: Le fichier n'a que {df.shape[1]} colonnes (min 16 requis)")
        return None, None

    colonne_11 = df.iloc[:, 10].values
    colonne_12 = df.iloc[:, 11].values
    y = df.iloc[:, 15].values

    if y.dtype == object or y.dtype == bool:
        y = y.astype(int)

    print("\n=== DONNÉES CHARGÉES ===")
    print(f"Total lignes: {len(y)}")

    # ================== CALCUL DES DIVISIONS ==================
    divisions_0, divisions_1 = [], []
    indices_0, indices_1 = [], []

    for i in range(len(y)):
        if colonne_12[i] != 0:
            div = colonne_11[i] / colonne_12[i]
            if y[i] == 0:
                divisions_0.append(div)
                indices_0.append(i)
            elif y[i] == 1:
                divisions_1.append(div)
                indices_1.append(i)

    divisions_0 = np.array(divisions_0)
    divisions_1 = np.array(divisions_1)

    # ================== FONCTION STATS ==================
    def afficher_stats(data, tag):
        if len(data) == 0:
            print(f"Aucune donnée valide pour TAG = {tag}")
            return None, None

        moyenne = np.mean(data)
        mediane = np.median(data)
        ecart_type = np.std(data)

        print(f"\n=== TAG = {tag} ===")
        print(f"Nombre de valeurs: {len(data)}")
        print(f"Moyenne: {moyenne:.6f}")
        print(f"Médiane: {mediane:.6f}")
        print(f"Écart-type: {ecart_type:.6f}")
        print(f"Min: {np.min(data):.6f}")
        print(f"Max: {np.max(data):.6f}")

        if ecart_type > 0:
            if abs(moyenne - mediane) / ecart_type < 0.1:
                print("→ Distribution symétrique")
            elif moyenne > mediane:
                print("→ Queue vers la droite")
            else:
                print("→ Queue vers la gauche")

        return moyenne, mediane

    # ================== STATS PAR GROUPE ==================
    moyenne_0, mediane_0 = afficher_stats(divisions_0, 0)
    moyenne_1, mediane_1 = afficher_stats(divisions_1, 1)

    # ================== STATS GLOBALES ==================
    divisions_totales = np.concatenate([divisions_0, divisions_1])
    print("\n=== GLOBAL ===")
    print(f"Divisions valides: {len(divisions_totales)}")
    print(f"Divisions par zéro ignorées: {len(y) - len(divisions_totales)}")
    print(f"Moyenne globale: {np.mean(divisions_totales):.6f}")
    print(f"Médiane globale: {np.median(divisions_totales):.6f}")

    if moyenne_0 and moyenne_1:
        print("\n=== COMPARAISON TAG 0 vs TAG 1 ===")
        print(f"Différence moyennes: {abs(moyenne_0 - moyenne_1):.6f}")
        print(f"Différence médianes: {abs(mediane_0 - mediane_1):.6f}")

    # ================== VISUALISATION ==================
    plt.figure(figsize=(12, 8))

    if len(divisions_0) > 0:
        plt.scatter(indices_0, divisions_0, alpha=0.5, s=25,
                    color='blue', label='Tag 0')

    if len(divisions_1) > 0:
        plt.scatter(indices_1, divisions_1, alpha=0.5, s=25,
                    color='red', label='Tag 1')

    if moyenne_0:
        plt.axhline(moyenne_0, color='darkblue', linewidth=3,
                    label=f'Moyenne 0 = {moyenne_0:.4f}')

    if moyenne_1:
        plt.axhline(moyenne_1, color='darkred', linewidth=3,
                    label=f'Moyenne 1 = {moyenne_1:.4f}')

    plt.xlabel("Index")
    plt.ylabel("Colonne11 / Colonne12")
    plt.title("Analyse statistique et visualisation des moyennes")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ================== RETOUR ==================
    X = np.column_stack((colonne_11, colonne_12))
    return X, y


if __name__ == "__main__":
    fichier_csv = "databaseDoS.csv"
    X, y = analyse_et_visualisation(fichier_csv)
