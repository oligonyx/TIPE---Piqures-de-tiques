# TIPE : Prévisions statistiques de signalements de piqûres de tiques


Les tiques constituent un réel problème de santé publique aujourd'hui : elles sont vectrices de multiples agents pathogènes, bien que toujours très peu connues du grand public. Être capable de prédire l'évolution des piqûres en France serait donc un atout en vue de prévenir les risques associés.


L'activité des tiques et le risque associé sont très variables, ce qui les rend complexes. Il serait donc intéressant de savoir si certains outils d'apprentissage automatique sont pertinents et efficaces pour prédire l'évolution des signalements de piqûres, et d'être capable d'estimer leur fiabilité.



## Scripts Python


### `make_sglmt_df.py`

Ce module sert à extraire les données de signalement puis à créer des tables `pandas.DataFrame` contenant le nombre de signalements par mois et par zone geographique.


### `create_maps.py`

Ce module sert à créer des cartes (exportées au format `.png`) qui comparent deux ensembles de signalements sur une ou plusieurs périodes.


### `predictions.py`

Ce module sert à créer un modèle `sklearn.tree.DecisionTreeClassifier` entraîné sur une periode. Il permet aussi de créer des cartes comparant des prédictions aux données réelles.


### `main.py`

Ce module sert à créer des modèles de prédictions de signalements et des cartes pour comparer les prédictions aux signalements réellement reçus.


