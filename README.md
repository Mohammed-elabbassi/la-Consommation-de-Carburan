# Prédiction de la consommation (mpg)

## Contexte et objectif
Le projet vise à construire un modèle de régression pour prédire la consommation de véhicules
(variable cible mpg, miles per gallon) à partir des caractéristiques du véhicule (cylindres, 
displacement, horsepower, weight, acceleration, model_year, origin). L’objectif est d’évaluer
plusieurs approches (régression linéaire, réseaux de neurones, XGBoost), comparer leurs 
performances et produire un modèle déployable via une application Flask.

## Données
•	Jeu de données utilisé : Auto MPG (fichier auto-mpg.data).
•	Colonnes importées :
["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name"]
•	Traitement initial :

o	Les valeurs manquantes sont représentées par ? dans le fichier d’origine — chargement avec na_values='?'.

o	Suppression des lignes contenant des valeurs manquantes (df.dropna()).

o	Suppression de la colonne texte car_name (non informative pour la régression directe).

## Prétraitement
•	Séparation variables/features / cible :
o	X = df.drop(columns=['mpg'])
o	y = df['mpg']
•	Split train/test : train_test_split(..., test_size=0.2, random_state=42) (80% apprentissage / 20% test).
•	Normalisation : StandardScaler() appliqué aux features ; scaler ajusté sur le jeu d'entraînement et appliqué
au test. Le scaler est conservé pour la mise en production.

## Modèles évalués
Dans le notebook principal, plusieurs modèles sont entraînés et comparés :
•	Régression linéaire multiple (LinearRegression()).
•	ELM / approximation via MLP 1 couche (MLPRegressor(hidden_layer_sizes=(50,), ...)) — un petit réseau dense.
•	BP (réseau de neurones profond) (MLPRegressor(hidden_layer_sizes=(100,50), ...)) — architecture plus profonde.
•	XGBoost (XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)).
Les modèles sont entraînés sur les mêmes données normalisées et évalués sur le jeu de test

## Métriques utilisées
•	RMSE (Root Mean Squared Error) — sensibilité aux grandes erreurs, bonne mesure d’écart absolu.
•	R² (coefficient de détermination) — proportion de variance expliquée.

## Résultats (résumé)
•	Les résultats sont présentés dans un DataFrame récapitulatif (RMSE et R² par modèle).
•	Le notebook trace des visualisations utiles :
o	Matrice de corrélation / heatmap entre features et mpg.
o	Distributions / boxplots éventuellement (exploration).
o	Courbe dispersion y_test vs y_pred pour le meilleur modèle (diag. d’identification d’erreurs systémiques).
•	Le meilleur modèle (celui avec le R² le plus élevé) est identifié et ses prédictions comparées aux valeurs
réelles sur un scatter plot avec la diagonale y=x en pointillés.
Remarque : les valeurs numériques exactes (RMSE et R²) dépendent du run et du jeu de données après nettoyage;
elles sont affichées dans le notebook et résument la performance comparative (le rapport inclut ces tableaux/plots).

## Modèle de production
•	Un notebook séparé (auto_voit.ipynb) prépare un MLPRegressor de grande capacité :
o	Architecture utilisée : hidden_layer_sizes=(256,128,64,32) (quatre couches cachées).
o	Paramètres : activation='relu', solver='adam', max_iter=2000, random_state=42 (selon notebook).
•	Après entraînement sur l’ensemble (après prétraitement et normalisation), le notebook sauvegarde le tuple
(model, scaler) dans model.pkl via pickle.dump((model, scaler), f).


## Déploiement 
•	Une application Flask (app.py) charge model.pkl (modèle + scaler) et expose une route /predict
(méthode POST) qui :
1.	Lit les valeurs du formulaire (request.form.values()), les convertit en floats.
2.	Applique scaler.transform([features]).
3.	Prédit mpg avec model.predict(...).
4.	Retourne prediction_text affichée dans le template result.html.
•	La page principale (/) renvoie index.html contenant le formulaire d’entrée.

## Interprétation & conclusion
•	L’exploration montre quelles caractéristiques influencent le plus le mpg (par corrélations
et importance selon XGBoost).
•	Les réseaux de neurones profonds (MLP à plusieurs couches) et XGBoost ont souvent de meilleures
performances que la régression linéaire sur ce type de données non linéaires, mais ils nécessitent 
plus de données/hyperparamétrage.
•	Pour la production : il est crucial de sauvegarder le scaler et d’appliquer exactement le même
prétraitement en production.


