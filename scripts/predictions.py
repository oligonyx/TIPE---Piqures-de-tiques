"""Predictions de signalements.

Ce module sert a creer un modele `sklearn.tree.DecisionTreeClassifier`
entraine sur une periode. Il permet aussi de creer des cartes comparant des
predictions aux donnes reelles.
"""


import os
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as mt
import joblib
from create_maps import comparative_total
from make_sglmt_df import sglmt_df



# charge le fichier contenant les constantes utiles
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'constants.json'), 'r') as cons:
    constants = json.load(cons)
# bordures de la carte
bounds = (constants["longitudeMin"], constants["longitudeMax"],
          constants["latitudeMin"], constants["latitudeMax"])
# donnees de signalements
sglmt_data_file = (constants["pathDonneesSignalements"],
                   constants["delimiteurDonneesSignalements"])
# donnees sur les regions
regions_file = (constants["pathDonneesRegions"],
                   constants["delimiteurDonneesRegions"])
# donnees sur les maladies de Lyme
with open(constants["pathDonneesLyme"], 'r') as lyme:
    lyme_data = (regions_file, json.load(lyme))
# repertoire ou stocker les modeles
model_path = constants["pathModele"]
# repertoire ou stocker les cartes
export_path = constants["pathCartes"]


def make_model(map_info, dates_info, model_name, lyme):
    """Cree un modele de prediction avec la methode `DecisionTreeClassifier`.
        `map_info`: triplet sur la construction des cartes
        `dates_info`: couple sur la periode d'entrainement 
        `model_name`: nom du modele a sauvegarder
        `lyme`: bool sur l'utilisation ou non des donnees de Lyme"""

    # utilisation de lyme
    lyme_info = lyme_data if lyme else None

    # creation de la dataframe avec le nb de sglmt par date et par zone
    df_train, _ = sglmt_df(bounds, sglmt_data_file,
                           map_info[:-1], dates_info, lyme_info)

    # donnees d'entrainement
    X_train = df_train.drop(columns='nb_sglmt')
    y_train = df_train[['nb_sglmt']].to_numpy(dtype=np.ushort).ravel()

    # creation, entrainement, sauvegarde du modele de prediction
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(model_path, model_name) + '.joblib')


def predict(map_info, dates_train, period, image_name, model_name, regions):
    """Predit des signalements a partir d'un modele, sauvegarde l'image de la
    comparaison avec les donnees reelles et renvoie le coefficient R2 global
    sur la correlation des donnees predites avec la theorie, et la liste des
    coefficients par periode.
        `map_info`: triplet sur la construction des cartes
        `dates_train`: couple sur la periode de test
        `period`: periode de temps sur laquelle on rassemble les signalements
        `image_name`: nom de l'image a creer 
        `model_name`: nom du modele a charger
        `lyme`: bool sur l'utilisation ou non des donnees de Lyme"""

    nb_div_long, nb_div_lat, _ = map_info

    # utilisation de lyme
    lyme_info = lyme_data if lyme else None

    # creation de la DataFrame avec le nb de sglmts par date et par zone
    df_test, dates_test = sglmt_df(bounds, sglmt_data_file,
                                   map_info[:-1], dates_train, lyme_info)

    # selection des donnees de test
    X_test = df_test.drop(columns='nb_sglmt')
    y_test = df_test[['nb_sglmt']].to_numpy(dtype=np.ushort)

    # chargement du modele et calcul des predictions
    model = joblib.load(os.path.join(model_path, model_name) + '.joblib')
    y_pred = model.predict(X_test)

    # formatage des donnees en sortie
    theory = np.hstack(y_test)
    predicted = y_pred.astype(dtype=np.ushort)

    # creation des cartes en fonction du mode choisi
    comparative_total(theory, predicted, map_info, dates_test, period,
                      (export_path, image_name))

    # coefficient de pearson pour evaluer la precision du modele
    theory2 = theory.reshape(len(dates_test), nb_div_lat * nb_div_long)
    predicted2 = predicted.reshape(len(dates_test), nb_div_lat * nb_div_long)
    main_r2_score = mt.r2_score(theory2, predicted2)
    r2_score_li = [(dates_test[i], mt.r2_score(theory2[i], predicted2[i]))
                   for i in range(len(dates_test))]

    return main_r2_score, r2_score_li

