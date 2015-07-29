# -*- coding: utf-8 -*-

from skimage.filters.rank import enhance_contrast
from skimage.external.tifffile import imread,imsave
from skimage.filters import gaussian_filter
from skimage.color import rgb2gray
from skimage.morphology import square
from skimage.measure import ransac
from skimage.transform import warp,AffineTransform
import cv2
import numpy as np
cimport numpy as np
import matplotlib.pyplot as plt
from os import listdir
from itertools import ifilter
from math import sqrt
from r import *
from cpython cimport bool


# Fonctions lambdas utiles
b = lambda x,tmp: tmp if tmp > x else x
c = lambda x,tmp: tmp if tmp < x else x

r = lambda k: np.where(k > 0)[0]
q = lambda k,v: r(k) if v.size == 0 else v
s = lambda k: k.size != 0
#s = lambda k: len(k) >= 200

# TODO:régler le problème du min/max, changer le facteur d'arrêt s...
o = lambda x,y: min(np.min(x),np.min(y)) #normalement max(np.min(x),np.min(y))
z = lambda x,y: max(np.max(x),np.max(y)) #normalement min(max,max)



def tracer(im):
    """ Cette fonction trace une image dans une nouvelle fenêtre.
    
    Args:
        im (np.ndarray): l'image en 8 bits.
    """

    plt.figure(figsize=(10,10))
    plt.imshow(im, cmap="gray", interpolation="nearest")
    plt.axis('off')
    plt.show()
    
def normaliser(im, seize=False):
    """ Cette fonction normalise les niveaux de gris d'une image, remet son max à 255 et les autres niveaux en conséquence.
    
    Args:
        im (np.ndarray): une image, en niveau de gris.

    Returns:
        L'image normalisée (np.ndarray).
    """
    mx = (1 << 8) - 1 if not seize else (1 << 16) - 1
    im = (mx/np.max(im))*im
    return im.astype(np.uint8) if not seize else im.astype(np.uint16)


cdef bonnesLignes(np.ndarray im):
    """ Cette fonction recherche les lignes extrèmes non-vides sur les images.
    Args:
        im: image dont on doit trouver les bonnes lignes. 
    Returns:
        Retourne le tableau des indexs non-nuls.
    """

    cdef np.ndarray x1 = np.zeros((0)),x2 = np.zeros((0)),y1 = np.zeros((0)),y2 = np.zeros((0))
    cdef int j=0

    while True:
        x1, y1 = q(im[:,j], x1), q(im[j,:], y1)
        x2, y2 = q(im[:, -(j+1)], x2), q(im[-(j+1),:], y2)

        if (s(x1) and s(x2) and s(y1) and s(y2)):
            break
        
        j+=1

    return o(x1,x2), z(x1,x2), o(y1,y2), z(y1,y2)






cdef class ParentRecEtPano:
    # private
    cdef float initial_sigma
    cdef object b, bf

    def __cinit__(self):
        self.initial_sigma = 1.6
        self.b = cv2.SURF()
        self.bf = cv2.BFMatcher()
   


    def sauvegardeInfosTif(self, pathIm, pathFichier=''):
        """ Utilise les fonctions de r.py pour récuperer et sauvegarder au besoin les données d'une image tif.
        Args:
            pathIm: le chemin de l'image.
            pathFichier: mettre un chemin pour sauvegarder.
        Returns:
            Le dico avec les données de la tif.
        """

        tmp = lectureDataTif(pathIm)

        if pathFichier != '':
            dataTifHDF5(tmp, pathFichier)

        return tmp
   


    cdef pointsInterets(self, np.ndarray imOne, np.ndarray imTwo):
        cdef np.ndarray im1,im2,d1,d2
        cdef list good, g1, g2, k1, k2, matches

        uno = lambda x: 255.*gaussian_filter(x, sqrt(self.initial_sigma**2 - 0.25))
        dos = lambda x: enhance_contrast(normaliser(x),square(5))

        im1,im2 = uno(imOne), uno(imTwo)
        im1,im2 = dos(im1), dos(im2)
        im1,im2 = normaliser(im1), normaliser(im2)
        
        k1,d1 = self.b.detectAndCompute(im1,None)
        k2,d2 = self.b.detectAndCompute(im2,None)
        
        matches = self.bf.knnMatch(d1,d2, k=2)

        # ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)
        
        g1,g2 = [],[]
        for i in good:
            g1.append(k1[i.queryIdx].pt)
            g2.append(k2[i.trainIdx].pt)

        return g1,g2
    

    cdef spc(self):
        pass

    cdef cropage(self):
        pass

    def run(self):
        self.spc()
        self.cropage()
     
    
cdef class AlignStack(ParentRecEtPano):

    # private
    cdef np.ndarray stack
    cdef int max_trials, minSetSize
    cdef float min_epsilon, max_epsilon, min_inlier_ratio

    
    # public
    cdef public np.ndarray stackAligne

    def __cinit__(self, *args):
        self.stack = self.choixDossierEtStack(args[0]).astype(float)
        self.stackAligne = np.zeros((0))
        
        self.max_trials = 10000
        self.min_epsilon = 2.
        self.max_epsilon = 100.
        self.min_inlier_ratio = 0.05
        self.minSetSize = 4

        ParentRecEtPano.__init__(self)



    def choixDossierEtStack(self, pathDossier):
        """ Cette fonction ouvre toutes les images d'un dossier et les met en stack.
        
        Args:
            pathDossier: le chemin du dossier contenant les images.
        
        Returns:
            Le stack d'images recalées dans le sens croissant des angles.
        """
        
        l = listdir(pathDossier)
        
        # filtrage, on ne garde que les tif + enlève fichiers cachés
        l = [x for x in l if x.endswith('.tif') and not x.startswith('.')]
        
        if len(l[0].split('=')) > 2:
            l = sorted(l, key=lambda x: float(x.split('=')[-1].split('d')[0]))
            #angles = sorted([float(x.split('=')[-1].split('d')[0]) for x in l])
        else: # pour images avec les heures, test
            l = list(ifilter(lambda x: len(x.split('-')) > 3, l))
            l = sorted(l, key=lambda i: float(i.split('-')[2].split('h')[0]))
        
        s = []

        # voir si lire info dans l'image puis hdf5 quelque part
        #self.sauvegardeInfosTif()
        
        for i in l:
            tmp = imread(pathDossier + '/' + i)
            tmp = tmp[:884,...]
            
            if tmp.shape[-1] == 3: # si rgb
                tmp = rgb2gray(tmp)
            
            s.append(normaliser(tmp))
            
        return np.array(s).astype(np.float)
        


    cdef cropage(self):
        """ Cette fonction recherche les 4 limites extrèmes sur les images du stack puis crope le stack et stocke le résultat dans stackAligne."""
        # Initialisation
        cdef unsigned int minx = 0, maxx = self.stack.shape[1], miny = 0, maxy = self.stack.shape[2]
        cdef int i = 0, lStack = len(self.stack)
        cdef unsigned int tmp_minx, tmp_maxx, tmp_miny, tmp_maxy

        # Calcul des bornes
        for i in xrange(lStack):
            tmp_minx, tmp_maxx, tmp_miny, tmp_maxy = bonnesLignes(self.stack[i,...])

            # On ne garde que les limites extrèmes du stack
            minx, miny, maxx, maxy = b(minx, tmp_minx), b(miny, tmp_miny), c(maxx,tmp_maxx), c(maxy, tmp_maxy)
           
        # Initialisation du stackAligne à la bonne taille
        self.stackAligne = np.zeros((lStack, maxx-minx, maxy-miny))

        # Cropage
        for i in xrange(lStack):
            self.stackAligne[i,...] = self.stack[i,minx:maxx, miny:maxy]
            
        self.stackAligne = self.stackAligne.astype(np.uint8)


    cdef spc(self): # recadrage
        """ Cette fonction recadre les images grâce à SURF et RANSAC."""
        cdef unsigned int lStack = len(self.stack) - 1, x = 0
        cdef object b = cv2.SURF(), bf = cv2.BFMatcher(), model
        cdef list g1,g2

        for x in xrange(lStack):
            print('Traitement image ' + str(x+1))

            g1, g2 = self.pointsInterets(self.stack[x,...], self.stack[x+1,...])

            model, _ = ransac((np.array(g1), np.array(g2)), AffineTransform, min_samples=3, residual_threshold=self.min_epsilon, max_trials=self.max_trials, stop_residuals_sum=self.min_inlier_ratio)
            
            self.stack[x+1,...] = warp(self.stack[x+1,...], AffineTransform(rotation=model.rotation, translation=model.translation), output_shape=self.stack[x+1].shape)



           
# TODO: Pano fonctionne pour deux images, tester pour 3+     
cdef class Pano(ParentRecEtPano):
    cdef list ims
    cdef public np.ndarray res

    def __cinit__(self, *args):
        # args contiendra les images à panoramiser
        ParentRecEtPano.__init__(self)

        self.ims = []
        for arg in args:
            self.ims.append(normaliser(arg))

        self.res = self.ims[0]
        
        
    cpdef spc(self): # pano
        cdef unsigned int x = 0
        cdef list g1, g2
        cdef np.ndarray H

        for x in xrange(len(self.ims) - 1):
            g2,g1 = self.pointsInterets(self.res, self.ims[x+1])
            H, _ = cv2.findHomography(np.array(g1),np.array(g2),8) # CV_Ransac = 8
            result = cv2.warpPerspective(self.ims[x+1], H, (self.res.shape[1] + self.ims[x+1].shape[1], self.ims[x].shape[0]))
            result[...,:self.ims[x].shape[1]] = self.res
            self.res = result
            self.cropage() 


    cpdef cropage(self):
        cdef unsigned int Minx, Maxx, Miny, Maxy
        cdef unsigned int minx = 0, maxx = self.res.shape[0], miny = 0, maxy = self.res.shape[1]

        Minx, Maxx, Miny, Maxy = bonnesLignes(self.res)
        self.res = self.res[Minx:Maxx, Miny:Maxy]


        
        
        
       
        
        
        
        
        
        

