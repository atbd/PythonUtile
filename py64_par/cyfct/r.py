# -*- coding: utf-8 -*-

import tables as t

def lectureDataTif(imagePath):
    """ Cette fonction lit les données d'une tif originale dans un dico.

    Args:
        imagePath (string): chemin jusqu'à la photo.

    Returns:
        Un dictionnaire contenant toutes les données de l'image.

    """

    tout,tmp = {},{}
    
    with open(imagePath, "rb") as f:
        for num,line in enumerate(reversed(f.readlines()),1):

            if line.startswith('[') and tmp != {}: # on a les catégories non-vides
                tout[line[1:].split(']')[0]],tmp = tmp,{}
                continue

            entry = line.strip().split('=', 2) 
            
            if len(entry) != 2: # si rien à enregistrer on termine cette itération
                continue
            
            tmp[entry[0]] = entry[1]

            if line.startswith('DataBar'):
                tmp[entry[0]] = tmp[entry[0]].split(' ')

            if num > 170: # en dehors des données à récup
                break

    return tout


def dataTifHDF5(dico, nomFichier):
    """ Sauvegarde les données contenue dans une tif originale dans un fichier hdf5.

    Args:
        dico (dict): le dictionnaire contenant les données, produit par lectureDataTif.
        nomFichier (string): le nom/chemin du fichier hdf5.
    """

    with t.openFile(nomFichier,'w', driver="H5FD_CORE") as f:
        f.createGroup('/', 'dataTif')
        for key,value in dico.iteritems(): # pour chaque catégorie, donne un autre dico
            f.createGroup("/dataTif", key) # création d'un dossier pour cette cat
            for k,v in value.iteritems():
                f.createArray("/dataTif/" + key + "/", k, v)


def sauvTableauHDF5(cheminFichier, nomTableau, tableau, dossier='', create=False, compression=False):
    """ Sauvegarde un tableau de données dans un dossier/sous-dossier dans un fichier hdf5.

    Args:
        cheminFichier (string): le chemin du fichier hdf5.
        nomTableau (string): le nom qu'aura le tableau dans l'hdf5.
        tableau (np.ndarray): le tableau de données.
        dossier (string): chemin du dossier dans l'hdf5, par exemple: data/aire/
        create (bool): si il faut créer les dossiers ou non, indiqué dans "dossier".
    """

    with t.openFile(cheminFichier, 'a', driver="H5FD_CORE") as f:
        if create == True:
            tmp = dossier.split('/')
            if len(tmp) > 1:
                if not f.root.__contains__(tmp[0]):
                    f.createGroup("/", tmp[0])
                for i in xrange(1, len(tmp)): # pour chaque dossier/sous-dossier
                    if not f.__contains__('/' + tmp[i-1] + '/' + tmp[i]):
                        f.createGroup("/" + tmp[i-1], tmp[i])
                        tmp[i] = tmp[i-1] + '/' + tmp[i]
            else:
                if not f.root.__contains__(dossier):
                    f.createGroup("/", dossier)

        if compression == True:
            f.create_carray("/" + dossier, nomTableau, obj=tableau ,filters=t.Filters(complib='zlib', complevel=5))
        else:
            f.createArray("/" + dossier, nomTableau, tableau)



