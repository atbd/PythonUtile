# -*- coding: utf-8 -*-

from skimage.filters.rank import enhance_contrast
from skimage.external.tifffile import imread,imsave
from skimage.filters import gaussian_filter
from skimage.color import rgb2gray
#from skimage.feature import (match_descriptors, corner_harris, corner_peaks, ORB, plot_matches, CENSURE)
from skimage.morphology import square
from skimage.measure import ransac
from skimage.transform import warp,AffineTransform
#from cv2 import warpAffine, getAffineTransform, estimateRigidTransform, findHomography, perspectiveTransform, warpPerspective
import cv2
import numpy as np
cimport numpy as np
import matplotlib.pyplot as plt
#from scipy.spatial.distance import euclidean
from os import listdir
from itertools import ifilter
from math import sqrt
from r import *
from cpython cimport bool
#from cython.parallel cimport parallel,prange
#import random as rnd
#from PointMatch import *
#from Model import *

def tracer(im):
    """ Cette fonction trace une image dans une nouvelle fenêtre.
    
    Args:
        im (np.ndarray): l'image en 8 bits.
    """

    plt.figure(figsize=(10,10))
    plt.imshow(im, cmap="gray", interpolation="nearest")
    plt.axis('off')
    plt.show()
    
cdef np.ndarray[np.uint8_t, ndim=2] normaliser(np.ndarray im):
    """ Cette fonction normalise les niveaux de gris d'une image, remet son max à 255 et les autres niveaux en conséquence.
    
    Args:
        im (np.ndarray): une image, en niveau de gris.

    Returns:
        L'image normalisée (np.ndarray).
    """

    im = (255./np.max(im))*im
    return im.astype(np.uint8)


cdef class AlignStack:

    # private
    cdef np.ndarray stack
    cdef int max_trials, minSetSize
    cdef float initial_sigma, min_epsilon, max_epsilon, min_inlier_ratio
    
    # public
    cdef public np.ndarray stackAligne

    def __cinit__(self, pathStack):
        self.stack = self.choixDossierEtStack(pathStack).astype(float)
        self.stackAligne = np.zeros((0))
        self.max_trials = 10000
        self.initial_sigma = 1.6
        self.min_epsilon = 2.
        self.max_epsilon = 100.
        self.min_inlier_ratio = 0.05
        self.minSetSize = 4


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


    def choixDossierEtStack(self, pathDossier):
        """ Cette fonction ouvre toutes les images d'un dossier et les met en stack.
        
        Args:
            pathDossier: le chemin du dossier contenant les images.
            tilt: pour le tri, si ce sont des images avec (tilt=xdeg) dans leurs noms.
        
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
        


    cdef np.ndarray[np.int64_t, ndim=1] rechercheBonneLigne(self, int i, bool rmin=True, bool rx=True):
        """ Cette fonction recherche les lignes extrèmes non-vides sur les images.
        Args:
            i: numero de l'image dans le stack.
            rmin: si True recherche le min sinon le max.
            rx: si True recherche les limites x sinon y.
        Returns:
            Retourne le tableau des indexs non-nuls.
        """
        cdef np.ndarray[np.float_t, ndim=2] tmp = self.stack[i,...]
        cdef np.ndarray[np.int64_t, ndim=1] r
        cdef int j = 0
        
        if rmin:
            if rx:
                while True:
                    r = np.where(tmp[:,j] > 0)[0]
                    if r.size != 0:
                        break
                    j+=1
            else:
                while True:
                    r=np.where(tmp[j,:] > 0)[0]
                    if r.size != 0:
                        break
                    j+=1
        else:
            if rx:
                while True:
                    r = np.where(tmp[:,-(j+1)] > 0)[0]
                    if r.size != 0:
                        break
                    j+=1
            else:
                while True:
                    r=np.where(tmp[-(j+1),:] > 0)[0]
                    if r.size != 0:
                        break
                    j+=1
        
        return r


    cdef cropage(self):
        """ Cette fonction recherche les 4 limites extrèmes sur les images du stack puis crope le stack et stocke le résultat dans stackAligne."""
        # Initialisation
        cdef unsigned int minx = 0, maxx = self.stack.shape[1], miny = 0, maxy = self.stack.shape[2]
        cdef int i = 0, lStack = len(self.stack)
        cdef unsigned int tmp_minx, tmp_maxx, tmp_miny, tmp_maxy
        cdef np.ndarray[np.int64_t, ndim=1] x1,x2,y1,y2

        # Calcul des bornes
        for i in xrange(lStack):
            x1 = self.rechercheBonneLigne(i, rmin=True, rx=True)
            x2 = self.rechercheBonneLigne(i, rmin=False, rx=True)
            y1 = self.rechercheBonneLigne(i, rmin=True, rx=False)
            y2 = self.rechercheBonneLigne(i, rmin=False, rx=False)

            tmp_minx, tmp_maxx, tmp_miny, tmp_maxy = min(np.min(x1), np.min(x2)), max(np.max(x1),np.max(x2)), min(np.min(y1), np.min(y2)), max(np.max(y1),np.max(y2))

            # On ne garde que les limites extrèmes du stack
            if tmp_minx > minx:
                minx = tmp_minx

            if tmp_miny > miny:
                miny = tmp_miny

            if tmp_maxx < maxx:
                maxx = tmp_maxx

            if tmp_maxy < maxy:
                maxy = tmp_maxy

        # Initialisation du stackAligne à la bonne taille
        self.stackAligne = np.zeros((lStack, maxx-minx, maxy-miny))

        # Cropage
        for i in xrange(lStack):
            self.stackAligne[i,...] = self.stack[i,minx:maxx, miny:maxy]
            
        self.stackAligne = self.stackAligne.astype(np.uint8)



    cdef recadrage(self):
        """ Cette fonction recadre les images grâce à SURF et RANSAC."""
        cdef unsigned int lStack = len(self.stack) - 1, x = 0
        cdef np.ndarray im1,im2,d1,d2
        cdef object b = cv2.SURF(), bf = cv2.BFMatcher(), model
        cdef list good, g1, g2, k1, k2, matches
        
        for x in xrange(lStack):
            print('Traitement image ' + str(x+1))
            
            im1,im2 = 255.*gaussian_filter(self.stack[x,...], sqrt(self.initial_sigma**2 - 0.25)), 255.*gaussian_filter(self.stack[x+1,...], sqrt(self.initial_sigma**2 - 0.25))
            im1,im2 = enhance_contrast(normaliser(im1), square(5)), enhance_contrast(normaliser(im2), square(5))
            im1,im2 = normaliser(im1), normaliser(im2)
            
            #b = cv2.SURF()
            #b = cv2.SIFT()
            
            k1,d1 = b.detectAndCompute(im1,None)
            k2,d2 = b.detectAndCompute(im2,None)
            
            #bf = cv2.BFMatcher()
            matches = bf.knnMatch(d1,d2, k=2)

            # Apply ratio test
            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append(m)
            
            g1,g2 = [],[]
            for i in good:
                g1.append(k1[i.queryIdx].pt)
                g2.append(k2[i.trainIdx].pt)

            model, _ = ransac((np.array(g1), np.array(g2)), AffineTransform, min_samples=3, residual_threshold=self.min_epsilon, max_trials=self.max_trials, stop_residuals_sum=self.min_inlier_ratio)
            
            self.stack[x+1,...] = warp(self.stack[x+1,...], AffineTransform(rotation=model.rotation, translation=model.translation), output_shape=self.stack[x+1].shape)


        
    cpdef run(self):
        self.recadrage()
        self.cropage()
        imsave("test.tif", self.stackAligne)
            
        
        
        
        
        


        
        
        
        

        
        
        
        
        
        
        
        
        
