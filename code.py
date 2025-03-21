import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Chargement des données
df = pd.read_csv("Dataset/dataset_etudiants.csv")

# Conversion des coéquipiers en interactions exploitables
interaction_data = []
for _, row in df.iterrows():
    etudiant_id = row['ID_Étudiant']
    try:
        coequipiers = eval(row['Coéquipiers'])  # Transformation de str -> list
        for c in coequipiers:
            interaction_data.append((etudiant_id, c, 1))  # 1 = interaction existante
    except:
        pass  # Gérer erreurs de format

# Création du DataFrame des interactions
interaction_df = pd.DataFrame(interaction_data, columns=['userID', 'itemID', 'rating'])

# Création de la matrice utilisateur-item
user_item_matrix = interaction_df.pivot(index='userID', columns='itemID', values='rating').fillna(0)

# Application du KNN
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(user_item_matrix)

def recommander_coequipiers(user_id, n_recommandations=3):
    if user_id not in user_item_matrix.index:
        return []
    distances, indices = knn.kneighbors([user_item_matrix.loc[user_id]], n_neighbors=n_recommandations+1)
    similar_users = indices.flatten()[1:]  # Exclure l'utilisateur lui-même
    
    recommandations = set()
    for u in similar_users:
        coequipiers = set(interaction_df[interaction_df['userID'] == user_item_matrix.index[u]]['itemID'])
        recommandations.update(coequipiers)
    
    recommandations -= set(interaction_df[interaction_df['userID'] == user_id]['itemID'])
    return list(recommandations)[:n_recommandations]

# Exemple d'utilisation
user_test = 1
print(f"Coéquipiers recommandés pour l'étudiant {user_test}: {recommander_coequipiers(user_test)}")

