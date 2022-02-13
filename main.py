"""Fichier principal pour les predictions.

Ce module sert a creer des modeles de predictions de signalements et des
cartes pour comparer les predictions aux signalements reellement recus.

Remplissez la partie "PARAMETRES" puis executez le fichier pour creer des
modeles et des cartes.

Remarques :
- Il est necessaire d'avoir rempli correctement le fichier `constants.json``
avant toute utilisation
- Les donnees de signalements sont disponibles jusqu'au mois d'octobre 2021
inclus, n'essayez donc pas d'entrainer un modele au-dela.
- Le "nombre de pixels par division" correspond au nombre de pixels pour les
cotes des divisions (qui sont des carres).
"""


from predictions import make_model, predict


##################
##  PARAMETRES  ##
##################

## Que faire ?

creer_un_modele = False
creer_des_cartes = True


## Carte

nombre_divisions_latitude = 120
nombre_divisions_longitude = 120
pixels_par_divion = 4


## Dates d'entrainement

annee_debut_entrainement = 17
mois_debut_entrainement = 4
annee_fin_entrainement = 20
mois_fin_entrainement = 10


## Dates de test

annee_debut_test = 18
mois_debut_test = 11
annee_fin_test = 21
mois_fin_test = 10


# Regroupement par periodes

nombre_mois_par_periode = 3


# Nom des fichiers

nom_modele = 'model75'

nom_image = 'img75'


##################


if __name__ == '__main__':

    map_info = (nombre_divisions_latitude,
                nombre_divisions_longitude,
                pixels_par_divion)

    dates_train = ((annee_debut_entrainement, mois_debut_entrainement),
                   (annee_fin_entrainement, mois_fin_entrainement))

    dates_test = ((annee_debut_test, mois_debut_test),
                   (annee_fin_test, mois_fin_test))

    if creer_un_modele:
        make_model(map_info, dates_train, nom_modele)

    if creer_des_cartes:
        r2_score = predict(map_info, dates_test,
                           nombre_mois_par_periode, nom_image, nom_modele)
        print(f'Score R\u00B2 des pr\u00E9disctions : {r2_score}')

