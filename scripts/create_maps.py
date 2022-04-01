"""Creation de cartes.

Ce module sert a creer des cartes (exportees au format `.png`) qui comparent
deux ensembles de signalements sur une ou plusieurs periodes.
"""


import numpy as np
from math import sqrt, ceil
from PIL import Image, ImagePalette, ImageDraw, ImageFont



def get_color_palette():
    """Renvoie une `ImagePalette` contenant les 102 couleurs necessaires :
    - de `0` a `100`: degrade blanc-violet
    - `101`: noir
    """

    # bytearray de la liste des triplets (R, G, B) des les couleurs utiles
    colors_bytes = (b'\xff\xff\xff\xae\x90\xed\x9d\x7f\xe3\x91u\xdc\x89m\xd5'
        +b'\x82g\xcf|b\xcav]\xc5rY\xc0mV\xbcjR\xb7fO\xb4cM\xb0`J\xac]H\xa9ZE'
        +b'\xa6XC\xa3VA\x9fS@\x9dQ>\x9aO<\x97M:\x94K9\x92I7\x8fH6\x8dF5\x8bD3'
        +b'\x88C2\x86A1\x84@0\x82>.\x80=-~<,|:+z9*x8)v7(t5\'s4&q3%o2$n1#l0#j/'
        +b'"i.!g- f,\x1fd+\x1fc*\x1ea)\x1d`(\x1c^\'\x1c]\'\x1b\\&\x1aZ%\x1aY$'
        +b'\x19X#\x18V#\x18U"\x17T!\x17S \x16Q \x15P\x1f\x15O\x1e\x14N\x1d'
        +b'\x14M\x1d\x13L\x1c\x12K\x1b\x12I\x1b\x11H\x1a\x11G\x19\x10F\x19'
        +b'\x10E\x18\x0fD\x17\x0fC\x17\x0eB\x16\x0eA\x16\r@\x15\r?\x14\x0c>'
        +b'\x14\x0c=\x13\x0b<\x13\x0b;\x12\x0b;\x12\n:\x11\n9\x11\t8\x10\t7'
        +b'\x0f\x086\x0f\x085\x0e\x084\x0e\x074\r\x073\r\x062\x0c\x061\x0c'
        +b'\x060\x0b\x05/\x0b\x05/\n\x04.\n\x04-\n\x04,\t\x03,\x00\x00\x00')

    return ImagePalette.ImagePalette(palette = colors_bytes)

# palette de couleurs restreinte aux couleurs utiles
palette = get_color_palette()


def get_square_like_grid(n):
    """Renvoie un couple `(height, width)` pour une disposition homogene de
    grille (similaire a un carre).
        `n`: nombre d'elements a placer"""

    height = ceil(sqrt(n))
    width = (n-1)//height + 1

    return (height, width)


def split_sglmt_sets(sglmts_li, nb_div_long, nb_div_lat, dates, period):
    """Decoupe puis somme les signalements par periode.
        `sglmts_li`: liste de tableaux de taille avec le nombre de
        signalements par date et par division
        `nb_div_long`: nombre de couches horizontales de la carte
        `nb_div_lat`: nombre de couches verticales de la carte
        `dates`: liste de couples `(annee, mois)`
        `period`: periode de temps (nombre de mois) sur laquelle on rassemble
        les signalements"""

    # decoupage des dates par periode
    period_nb, remainder = divmod(len(dates),period)
    dates_split = [(dates[period*i], dates[period*(i+1)-1])
                   for i in range(period_nb)]
    dates = dates_split + ([] if remainder == 0
                              else [(dates[len(dates)-remainder], dates[-1])])

    # cas du reste non nul
    if remainder > 0:
        remainder_ar = np.zeros((period-remainder)*nb_div_long*nb_div_lat,
                                dtype=np.ushort)
        for i in range(len(sglmts_li)):
            sglmts_li[i] = np.concatenate((sglmts_li[i], remainder_ar))

    # decoupage des tableaux
    for i in range(len(sglmts_li)):
        sglmts_split = np.hsplit(sglmts_li[i], len(dates))
        sglmts_li[i] = [np.sum(np.hsplit(sglmts, period), axis=0,
                               dtype=np.ushort)
                        for sglmts in sglmts_split]

    return sglmts_li, dates


def make_single_map(nb_sglmts, nb_div_long, nb_div_lat, max_val):
    """Cree une image (carte de la France) et y place un ensemble de
    signalements.
        `nb_sglmts`: tableau de taille `nb_div_long*nb_div_lat` avec nb de
        signalements par division
        `nb_div_long`: nombre de couches horizontales de la carte
        `nb_div_lat`: nombre de couches verticales de la carte
        `max_val`: nombre maximal de signalements en un point"""

    # calcul des couleurs associees
    pixels = (nb_sglmts*100/max_val).astype(dtype=np.ubyte)
    pixels = pixels.reshape(nb_div_lat, nb_div_long)

    # creation de l'image
    map = Image.fromarray(obj=pixels[::-1], mode='P')
    map.putpalette(palette)

    return map


def get_legends_li(dates, period):
    """Renvoie la liste des legendes (mois sur lesquels on predit) pour
    l'image finale.
        `dates`: liste de couples `(debut, fin)` sur chaque periode etudiee
        `period`: periode de temps (nombre de mois) sur laquelle on rassemble
        les signalements"""

    if period == 1:
        return [f'    {date[1]:0>2} / 20{date[0]}' for (date,_) in dates]
    else:
        return [f'{start[1]:0>2} / 20{start[0]}   -'
                +f'   {end[1]:0>2} / 20{end[0]}'
                for (start, end) in dates]


def get_all_maps(sglmts1, sglmts2, map_info, dates, period):
    """Renvoie des listes des cartes a placer, le decoupage de l'image en
    cartes et la liste des dates associee.
        `sglmts1` & `sglmts2`: tableaux avec le nombre de signalements par
        date et par division
        `map_info`: triplet `(nb_div_long, nb_div_lat)` sur la construction
        des cartes 
        `dates`: liste de couples `(annee, mois)`
        `period`: periode de temps (nombre de mois) sur laquelle on rassemble
        les signalements"""

    nb_div_long, nb_div_lat = map_info

    # decoupage des dates et des tableaux
    sglmts, dates = split_sglmt_sets(
        [sglmts1, sglmts2], nb_div_long, nb_div_lat, dates, period)
    sglmts1, sglmts2 = sglmts[0], sglmts[1]

    # repartition des cartes en grille
    grid_height, grid_width = get_square_like_grid(len(dates))
    # valeur maximale au total
    max_val = np.amax(sglmts)
    # creation et ajout des sous-cartes aux listes
    map_li1, map_li2 = [], []
    for grid_x in range(grid_width):
        for grid_y in range(min(grid_height, len(dates)-grid_x*grid_height)):
            i = grid_x*grid_height + grid_y
            map_li1 += [make_single_map(sglmts1[i],
                                        nb_div_long, nb_div_lat, max_val)]
            map_li2 += [make_single_map(sglmts2[i],
                                        nb_div_long, nb_div_lat, max_val)]

    return [map_li1, map_li2], (grid_height, grid_width), dates


def comparative_total(sglmts1, sglmts2, map_info, dates, period, export_info):
    """Comparaison de deux ensemble de cartes accumulant tous les signalements
    pour chaque periode de temps.
        `sglmts1` & `sglmts2`: tableaux de taille avec le nombre de
        signalements par date et par division
        `map_info`: triplet `(nb_div_long, nb_div_lat, div_size)` sur la
        construction des cartes
        `dates`: liste de couples `(annee, mois)`
        `period`: periode de temps (nombre de mois) sur laquelle on rassemble
        les signalements
        `export_info`: couple `(export_path, file_name)` sur le fichier a
        creer"""

    nb_div_long, nb_div_lat, div_size = map_info
    export_path, file_name = export_info

    # cartes, grille et dates
    [map_li1, map_li2], (grid_height, grid_width), dates = get_all_maps(
        sglmts1, sglmts2, map_info[:-1], dates, period)
    nb_prds = len(dates)

    # organisation spatiale : bordures, cartes, legendes, titres, decalage
    sml_brd, lrg_brd = 2, 6
    map_width, map_height = nb_div_long+sml_brd, nb_div_lat+sml_brd
    lgnd_height, title_height = 16, 28
    width = 2*(grid_width*map_width-sml_brd) + lrg_brd
    height = grid_height*(map_height+lgnd_height) - sml_brd + title_height
    x_shift = grid_width*map_width - sml_brd + lrg_brd

    # carte (reduite)
    reduced_map = Image.new(mode='P', size=(width, height), color=101)
    reduced_map.putpalette(palette)

    for grid_x in range(grid_width):
        for grid_y in range(min(grid_height, nb_prds-grid_x*grid_height)):
            i = grid_x*grid_height + grid_y
            x_map, y_map = grid_x*map_width, grid_y*(map_height+lgnd_height)
            # ajout des sous-cartes
            reduced_map.paste(map_li1[i], (x_map,y_map))
            reduced_map.paste(map_li2[i], (x_map+x_shift,y_map))
            # rectangles pour la legende
            lgnds_red = ImageDraw.Draw(reduced_map)
            y_lgnd1 = y_map+nb_div_lat
            x_lgnd2, y_lgnd2 = x_map+nb_div_long-1, y_lgnd1+lgnd_height-1
            coord1 = (x_map, y_lgnd1, x_lgnd2, y_lgnd2)
            coord2 = (x_map+x_shift, y_lgnd1, x_lgnd2+x_shift, y_lgnd2)
            lgnds_red.rectangle(coord1, fill=50)
            lgnds_red.rectangle(coord2, fill=50)

    # carte finale
    full_width = width*div_size
    full_height = height*div_size
    full_x_shift = x_shift*div_size
    map = reduced_map.resize((full_width, full_height),
                             resample=Image.NEAREST)
    font = ImageFont.truetype('cmunrm.ttf', 10*div_size)

    # legendes des cartes
    lgnds = ImageDraw.Draw(map)
    legends_li = get_legends_li(dates, period)
    for grid_x in range(grid_width):
        for grid_y in range(min(grid_height, nb_prds-grid_x*grid_height)):
            i = grid_x*grid_height + grid_y
            x_lgnd1 = (grid_x*map_width+map_width//6)*div_size
            y_lgnd1 = (grid_y*(map_height+lgnd_height)+nb_div_lat+2)*div_size
            coord1 = (x_lgnd1, y_lgnd1)
            coord2 = (x_lgnd1+full_x_shift, y_lgnd1)
            lgnds.text(coord1, legends_li[i], fill=0, font=font)
            lgnds.text(coord2, legends_li[i], fill=0, font=font)

    # titres 
    x_ttl1 = full_width//6+10*div_size
    y_ttl1 = (grid_height*(map_height+lgnd_height)-sml_brd+4)*div_size
    coord1, coord2 = (x_ttl1, y_ttl1), (x_ttl1+full_x_shift, y_ttl1)
    font_titles = ImageFont.truetype('cmunrm.ttf', 16*div_size)
    lgnds.text(coord1, 'REÇUS', fill=0, font=font_titles)
    lgnds.text(coord2, 'PRÉDITS', fill=0, font=font_titles)
    # sauvegarde de la carte
    map.save(export_path + file_name + '.png')



