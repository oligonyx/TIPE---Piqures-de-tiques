"""Creation de dataframes.

Ce module sert a extraire les donnees de signalement puis à creer des tables
`pandas.DataFrame` contenant le nombre de signalements par mois et par
zone geographique.
"""


import numpy as np
import pandas as pd



def get_dates_array(rel_months):
    """Cree un tableau avec les annees et les mois d'une date a une autre.
        `rel_months`: premier et dernier mois relatif.
        (numero du mois depuis janvier 2000) etudie"""

    months_array = np.arange(rel_months[0]-1, rel_months[1])
    years, months = divmod(months_array, 12)
    dates = np.stack([years, months+1], axis=1)

    return dates


def sglmt_data_simpli(bounds, source_file, map_info, rel_months):
    """Importe le fichier `.csv` des signalements et renvoie une version
    discretisee en `pandas.DataFrame`.
        `bounds`: quadruplet `(long_min, long_max, lat_min, lat_max)`
        sur les bordures des cartes
        `source_file`: couple `(data_path, delim)` sur le fichier de
        signalements
        `map_info`: couple `(nb_div_long, nb_div_lat)` sur la construction
        des cartes
        `map_info`: couple sur le premier et dernier mois relatif etudie"""

    data_path, data_delim = source_file
    long_min, long_max, lat_min, lat_max = bounds
    nb_div_long, nb_div_lat = map_info
    rel_month_min, rel_month_max = rel_months

    # chargement du fichier
    sglmt_data = pd.read_csv(data_path, delimiter=data_delim)
    sglmt_data = sglmt_data.drop(columns=['ref', 'day'])

    # simplification des dates en mois relatifs
    sglmt_data['year'] -= 2000
    sglmt_data['rel_month'] = sglmt_data['year']*12 + sglmt_data['month']
    sglmt_data = sglmt_data.drop(columns=['month', 'year'])
    sglmt_data['rel_month'] = sglmt_data['rel_month'].astype(int)

    # selection des donnees utiles uniquement
    sglmt_data = sglmt_data[(sglmt_data['rel_month'] >= rel_month_min)
                            & (sglmt_data['rel_month'] <= rel_month_max)]

    # discretisation des coordonnees gps
    lat_unit = nb_div_lat/(lat_max - lat_min)
    sglmt_data['lat'] =  (sglmt_data['lat'] - lat_min) * lat_unit
    sglmt_data['lat'] = sglmt_data['lat'].apply(np.floor)
    sglmt_data['lat'] = sglmt_data['lat'].astype(int)

    long_unit = nb_div_long/(long_max - long_min)
    sglmt_data['long'] = (sglmt_data['long'] - long_min) * long_unit
    sglmt_data['long'] = sglmt_data['long'].apply(np.floor)
    sglmt_data['long'] = sglmt_data['long'].astype(int)

    # indice des combinaisons de mois et de zone
    sglmt_data['indices'] = ((sglmt_data['rel_month']-rel_month_min)
                             * nb_div_lat*nb_div_long
                             + sglmt_data['lat']*nb_div_long
                             + sglmt_data['long'])

    # compte des signalements par indice
    sglmt_data = ((pd.DataFrame(sglmt_data['indices'], columns=['indices']))
                  .pivot_table(columns=['indices'], aggfunc='size'))
    sglmt_data = pd.DataFrame(sglmt_data, columns=['nb_sglmt'])

    return sglmt_data


def sglmt_df(bounds, source_file, map_info, dates_info):
    """Cree une `pandas.DataFrame` avec tous les signalements sur une periode
    par mois et par division.
        `bounds`: quadruplet sur les bordures des cartes
        `source_file`: couple sur le fichier de signalements
        `map_info`: couple sur la construction des cartes
        `dates_info`: couple sur la periode etudiee"""

    nb_div_long, nb_div_lat = map_info
    year_min, month_min = dates_info[0]
    year_max, month_max = dates_info[1]

    rel_months = year_min*12+month_min, year_max*12+month_max

    # simplification des signalements
    base = sglmt_data_simpli(bounds, source_file, map_info, rel_months)

    # tableau des dates sur la periode
    dates = get_dates_array(rel_months)
    nb_dates = len(dates)

    # creation de la DataFrame avec chaque combinaison de date et de zone
    map_size = nb_div_lat*nb_div_long
    df = pd.DataFrame()
    df[['year', 'month']] = np.repeat(dates, map_size, axis=0)
    df['lat'] = np.tile(np.repeat(np.arange(nb_div_lat, dtype=np.ushort),
                                  nb_div_long), nb_dates)
    df['long'] = np.tile(np.arange(nb_div_long, dtype=np.ushort),
                         nb_dates*nb_div_lat)
    df['nb_sglmt'] = np.zeros(nb_dates*map_size, dtype=np.ushort)

    # ajout de tous les signalements
    for row in base.iterrows():
        df.at[row[0], 'nb_sglmt'] = row[1]

    return df, dates

