"""Fichier principal pour les predictions.

Ce module sert a creer des modeles de predictions de signalements et des
cartes pour comparer les predictions au signalements rellement recus.

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


from scripts.predictions import make_model, predict


##################
##  PARAMETRES  ##
##################

## Que faire ?

creer_un_modele = False
creer_des_cartes = False
utiliser_lyme = False


## Carte

nombre_divisions_latitude = 120
nombre_divisions_longitude = 120
pixels_par_divion = 4


## Dates d'entrainement

mois_debut_entrainement = 4
annee_debut_entrainement = 17
mois_fin_entrainement = 10
annee_fin_entrainement = 20


## Dates de test

mois_debut_test = 11
annee_debut_test = 20
mois_fin_test = 10
annee_fin_test = 21


## Regroupement par periodes

nombre_mois_par_periode = 1


## Nom des fichiers

nom_modele = (f'model({mois_debut_entrainement:0>2}-{annee_debut_entrainement}'
              +f'->{mois_fin_entrainement:0>2}-{annee_fin_entrainement})')

nom_image = (f'pred({mois_debut_entrainement:0>2}-{annee_debut_entrainement}'
             +f'->{mois_fin_entrainement:0>2}-{annee_fin_entrainement})'
             +f'_test({mois_debut_test:0>2}-{annee_debut_test}'
             +f'->{mois_fin_test:0>2}-{annee_fin_test})')



###############################################
##  EXECUTION - NE RIEN MODIFIER CI-DESSOUS  ##
###############################################

if __name__ == '__main__':
    map_info = (nombre_divisions_latitude,
                nombre_divisions_longitude,
                pixels_par_divion)

    dates_train = ((annee_debut_entrainement, mois_debut_entrainement),
                   (annee_fin_entrainement, mois_fin_entrainement))

    dates_test = ((annee_debut_test, mois_debut_test),
                   (annee_fin_test, mois_fin_test))

    if creer_un_modele:
        make_model(map_info, dates_train, nom_modele, utiliser_lyme)

    if creer_des_cartes:
        r2_score, r2_score_li = predict(
            map_info, dates_test, nombre_mois_par_periode, nom_image,
            nom_modele, utiliser_lyme)
        
        score_li = ''
        for date, score in r2_score_li:
            score_li += f'\n\u2000 {date[1]:0>2}/20{date[0]} : {score}'
        
        print(f'Score R\u00B2 global des pr\u00E9dictions : {r2_score}\n'
              f'Score R\u00B2 par p\u00E9riode :{score_li}')
