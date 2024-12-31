import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.decomposition import FastICA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# Chargement des données
data = pd.read_csv('data/diabetes_prediction_dataset.csv')

# Filtrage des données
data = data[data['gender'] != 'Other']  
data = data[data['bmi'] <= 60]

# Prétraitement des données
data['smoking_history'] = data['smoking_history'].replace(['ever'], 'former')
data['smoking_history'] = data['smoking_history'].replace(['not current'], 'never')
data['age'] = pd.cut(data['age'], bins=[0, 16, 32, 48, 64, 80], labels=['0-16', '17-32', '33-48', '49-64', '65-80']) 
data['bmi'] = pd.cut(data['bmi'], bins=[10, 20, 30, 40, 50, 60], labels=['10-20', '20-30', '30-40', '40-50', '50-60'])

# Encodage des données catégorielles
gender_encoder = LabelEncoder()
smoking_encoder = LabelEncoder()
age_encoder = LabelEncoder()
bmi_encoder = LabelEncoder()
data['gender'] = gender_encoder.fit_transform(data['gender'])
data['smoking_history'] = smoking_encoder.fit_transform(data['smoking_history'])
data['age'] = age_encoder.fit_transform(data['age'])
data['bmi'] = bmi_encoder.fit_transform(data['bmi'])
with open('pkl/label_encoders.pkl', 'wb') as file:
    pickle.dump({'gender': gender_encoder, 'age': age_encoder, 'smoking_history': smoking_encoder, 'bmi': bmi_encoder}, file)

# Réduction de la dimensionnalité
ica = FastICA(n_components=1)
data['ica_glucose'] = ica.fit_transform(data[['blood_glucose_level', 'HbA1c_level']])
data = data.drop(columns=['blood_glucose_level', 'HbA1c_level'])

# Sélection des caractéristiques
X = data[data.columns[data.columns != 'diabetes']]
y = data['diabetes']

# Standardisation des données
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
with open('pkl/scaler_file.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Définition des paramètres et des valeurs à tester
models= {
'RandomForestClassifier' : {
'model' : RandomForestClassifier(random_state=42),
'params': {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
},
'LogisticRegression': {
'model': LogisticRegression(random_state=42),
'params': {
'C':[0.1, 1, 10, 100],
'solver': [ 'newton-cg' , 'lbfgs' , 'liblinear'],
'max_iter': [100, 200, 300]
}
},
'DecisionTreeClassifier': {
'model': DecisionTreeClassifier(random_state=42),
'params': {
'criterion': ['gini', 'entropy'],
'max_depth': [None, 10, 20],
'min_samples_split': [2, 5, 10],
'min_samples_leaf': [1, 2, 4]
}
},
'KNeighborsClassifier': {
'model': KNeighborsClassifier(),
'params': {
'n_neighbors': [3, 5, 7, 9],
'weights': ['uniform', 'distance'],
}
},
'SVC': {
    'model': SVC(),
    'params': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['rbf'],
        'gamma': ['scale', 'auto']
}
}
}


# Définition du "meilleur modèle"
best_model={}

# Définition dude précision pour chaque modèle
model_accuracies={}

# Boucles d'entrainement de chaque modèle et d'optimisation
for model_name, model_params in models.items():
    print(f"Entrainement et optimisation du modèle : {model_name}")
    grid_search = GridSearchCV(model_params['model'], model_params['params'], cv=5, n_jobs = -1 , verbose =3)
    grid_search.fit(X_train, y_train)

    # Stocke le meilleur modèle
    best_model[model_name] = grid_search.best_estimator_

    # Calcule la précision sur le jeu de test
    test_accuracy = accuracy_score(y_test, best_model[model_name].predict(X_test))

    # Ajoute la précision calculée au dictionnaire des précisions
    model_accuracies[model_name] = test_accuracy

    # Affiche les meilleurs paramètres
    print(f"Meilleurs paramètres pour {model_name} : {grid_search.best_params_}")
    print(f"Précision sur l'ensemble de test pour {model_name} : {test_accuracy:.4f}")

# Récapitulatif des précisions pour tous les modèles
print("\n--- Précisions des modèles optimisés ---")
for model_name, accuracy in model_accuracies.items():
    print(f"{model_name}: Précision sur l'ensemble de test = {accuracy:.4f}")

# Sélection des quatre meilleurs modèles 
sorted_models = sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True)
top_4_models = dict(sorted_models[:4])

print("\n--- Quatre meilleurs modèles ---")
for model_name, accuracy in top_4_models.items():
    print(f"{model_name}: Précision = {accuracy:.4f}")
    # Sauvegarde des quatre meilleurs modèles
    with open(f'pkl/{model_name}_model.pkl', 'wb') as modele_file:
        pickle.dump(best_model[model_name], modele_file)

# Création des courbes d'apprentissage
plt.figure(figsize=(20, 10))

for idx, (model_name, _) in enumerate(top_4_models.items(), 1):
    plt.subplot(2, 2, idx)
    
    train_sizes, train_scores, val_scores = learning_curve(
        best_model[model_name], X_scaled, y, 
        cv=5, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.plot(train_sizes, train_mean, label='Score entraînement', color='blue')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, label='Score validation', color='red')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.title(f'Courbe d\'apprentissage - {model_name}')
    plt.xlabel('Taille de l\'ensemble d\'entraînement')
    plt.ylabel('Score')
    plt.legend(loc='lower right')
    plt.grid(True)

plt.tight_layout()
plt.show()

# Tracer les matrices de confusion pour les quatre meilleurs modèles
plt.figure(figsize=(15, 10))
for idx, (model_name, _) in enumerate(top_4_models.items()):
    model = best_model[model_name]
    cm = confusion_matrix(y_test, model.predict(X_test))
    
    plt.subplot(2, 2, idx + 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={'size': 24})
    plt.title(f'Matrice de confusion - {model_name}')
    plt.xlabel('Prédiction')
    plt.ylabel('Véridique')

plt.tight_layout()
plt.show()

# Affichage des importances des variables pour le meilleur modèle
best_model_name = list(top_4_models.keys())[0]  # Récupère le nom du meilleur modèle
best_classifier = best_model[best_model_name]

# Vérification si le modèle a l'attribut feature_importances_
if hasattr(best_classifier, 'feature_importances_'):
    # Création d'un DataFrame avec les noms des variables et leurs importances
    feature_importance = pd.DataFrame({
        'Variable': X.columns,
        'Importance': best_classifier.feature_importances_
    })
    
    # Tri par importance décroissante
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    print(f"\nImportance des variables pour {best_model_name}:")
    # Affichage des importances de chaque variable
    for index, row in feature_importance.iterrows():
        print(f"{row['Variable']}: {row['Importance']:.4f}")