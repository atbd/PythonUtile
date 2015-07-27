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
import matplotlib.pyplot as plt
#from scipy.spatial.distance import euclidean
from os import listdir
from itertools import ifilter
from math import sqrt
from r import *
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
    
def normaliser(im):
    """ Cette fonction normalise les niveaux de gris d'une image, remet son max à 255 et les autres niveaux en conséquence.
    
    Args:
        im (np.ndarray): une image, en niveau de gris.

    Returns:
        L'image normalisée (np.ndarray).
    """

    im = (255.0/np.max(im))*im
    return im.astype(np.uint8)

# pour avoir les z de l'image tif 16 bits en fct du min mi et max ma en nm
#lut = [(((i*ma) + (2**16-1 - i)*mi)/(2**16-1)) for i in xrange(2**16)]

class AlignStack:
    """ Attributs:
    """

    def __init__(self, pathStack):
        self.stack = self.choixDossierEtStack(pathStack).astype(float)
        self.stackAligne = np.zeros((0))
        self.max_trials = 10000
        self.initial_sigma = 1.6
        self.min_epsilon = 2.
        self.max_epsilon = 100.
        self.min_inlier_ratio = 0.05
        self.minSetSize = 4


    """
    def rotEtTrans(self,kp1,kp2):
        # kp{1,2} ont 2 couples de points.
        
        a,b = (kp1[0][0] - kp1[1][0]), (kp1[0][1] - kp1[1][1])
        t = asin((a*(kp2[0][1] - kp2[1][1]) - b*(kp2[0][0] - kp2[1][0])) / (a**2 + b**2)) # t en radians

        tx = kp2[1][0] - kp1[1][0]*cos(t) + kp1[1][1]*sin(t)
        ty = kp2[1][1] - kp1[1][0]*sin(t) - kp1[1][1]*cos(t)

        return t,tx,ty
    """
    
    '''
    def trouverEtMatchKp(self, im1, im2, nbPts):
        """ Cette fonction trouve et match les keypoints ensemble grâce à ORB.
        Args:
            im1: première image ou chercher les keypoints.
            im2: seconde image ou chercher les keypoints.
            nbPts: le nombre de keypoints à trouver.

        Returns:
            Les keypoints matchés entre eux, k1 et k2 pour l'image 1 et 2 respectivement.
        """
        # utilisation de ORB
        descriptor_extractor = ORB(n_keypoints=nbPts)    
        
        descriptor_extractor.detect_and_extract(im1)
        kp1 = descriptor_extractor.keypoints # les kp sont des np.ndarray
        d1 = descriptor_extractor.descriptors
        
        descriptor_extractor.detect_and_extract(im2)
        kp2 = descriptor_extractor.keypoints
        d2 = descriptor_extractor.descriptors
        
        print('===============================')
        print(str(len(kp2)) + ' points trouvés.')
        
        # donne un np.array avec les indices correspondants
        matches = match_descriptors(d1, d2, cross_check=True, max_distance=self.max_epsilon)

        print(str(len(matches)) + ' correspondances.')

        # On ne garde que les kp qui match ensemble
        k1,k2 = [],[]
        for x in matches:
            k1.append(kp1[x[0]])
            k2.append(kp2[x[1]])
            #k1.append(np.array(list(kp1[x[0]]) + [0.]))
            #k2.append(np.array(list(kp2[x[1]]) + [0.]))

        return np.array(k1),np.array(k2)
        '''


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


    def choixDossierEtStack(self, pathDossier): #, tilt=True):
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
        
        #if tilt:
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
            
        return np.array(s)
        


    def rechercheBonneLigne(self, i, rmin=True, rx=True):
        """ Cette fonction recherche les lignes extrèmes non-vides sur les images.
        Args:
            i: numero de l'image dans le stack.
            rmin: si True recherche le min sinon le max.
            rx: si True recherche les limites x sinon y.
        Returns:
            Retourne le tableau des indexs non-nuls.
        """
        tmp = self.stack[i,...]
        j = 0
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


    def cropage(self):
        """ Cette fonction recherche les 4 limites extrèmes sur les images du stack puis crope le stack et stocke le résultat dans stackAligne."""
        # Initialisation
        minx, maxx, miny, maxy = 0, self.stack.shape[1], 0, self.stack.shape[2]

        # Calcul des bornes
        for i in xrange(len(self.stack)):
            x1 = self.rechercheBonneLigne(i)
            x2 = self.rechercheBonneLigne(i, rmin=False)
            y1 = self.rechercheBonneLigne(i, rx=False)
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
                
        print('minx: ' + str(minx))
        print('maxx: ' + str(maxx))
        print('miny: ' + str(miny))
        print('maxy: ' + str(maxy))

        # Initialisation du stackAligne à la bonne taille
        self.stackAligne = np.zeros((len(self.stack), maxx-minx, maxy-miny))

        # Cropage
        for i in xrange(len(self.stack)):
            self.stackAligne[i,...] = self.stack[i,minx:maxx, miny:maxy]
            
        self.stackAligne = self.stackAligne.astype(np.uint8)
        

    '''
    def run(self):
        """ Cette fonction recadre les images grâce à ORB. Ne fonctionne pas très bien."""
        for x in xrange(1):#xrange(len(self.stack)-1):
            # rebouclage
            im1,im2 = 255.*gaussian_filter(self.stack[x,...], sqrt(self.initial_sigma**2 - 0.25)), 255.*gaussian_filter(self.stack[x+1,...], sqrt(self.initial_sigma**2 - 0.25))
            #im1,im2 = enhance_contrast(normaliser(im1), square(3)), enhance_contrast(normaliser(im2), square(3))
            # recherche des correspondants
            kp1,kp2 = self.trouverEtMatchKp(im1, im2, 2500)

            """
            # test
            k1,k2 = [], []
            for x in xrange(len(kp1)):
                k1.append(Point(kp1[x]))
                k2.append(Point(kp2[x]))

            pm = []
            for y in xrange(len(k1)-1):
                pm.append(PointMatch(k1[y], k2[y]))

            inliers = []
            model = TRModel2D()
            model = model.estimateBestModel(pm, inliers, self.min_epsilon, self.max_epsilon, self.min_inlier_ratio)
            """




            # inliers par RANSAC
            model, inliers = ransac((kp2, kp1), AffineTransform, min_samples=3, residual_threshold=self.min_epsilon, max_trials=1500, stop_residuals_sum=self.min_inlier_ratio)
            #model = self.estimateBestModel(kp1,kp2)
            
            t =tuple(map(float, model.translation))
            r = float(model.rotation)

            print('Rotation: ' + str((180./3.14159)*r) + ' degrées.')
            print('Translation: ' + str(t))
            print('===============================')

            
            self.stack[x+1] = warp(self.stack[x+1],AffineTransform(rotation=r, translation=t),output_shape=self.stack[x+1].shape)
            
        #return self.stack
        #return model
        #return kp1,kp2
        '''


    def run2(self):
        """ Cette fonction recadre les images grâce à SIFT et RANSAC, fonctionne bien."""
        for x in xrange(len(self.stack)-1):
            print('Traitement image ' + str(x+1))
            im1,im2 = 255.*gaussian_filter(self.stack[x,...], sqrt(self.initial_sigma**2 - 0.25)), 255.*gaussian_filter(self.stack[x+1,...], sqrt(self.initial_sigma**2 - 0.25))
            im1,im2 = enhance_contrast(normaliser(im1), square(3)), enhance_contrast(normaliser(im2), square(3))
            im1, im2 = normaliser(im1), normaliser(im2)

            sift = cv2.SIFT()

            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(im1,None)
            kp2, des2 = sift.detectAndCompute(im2,None)

            # BFMatcher with default params
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1,des2, k=2)

            # Apply ratio test
            good = []
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append(m)


            # good est une liste de DMatch
            g1,g2 = [],[]
            for i in good:
                g1.append(kp1[i.queryIdx].pt)
                g2.append(kp2[i.trainIdx].pt)

            model, inliers = ransac((np.array(g1), np.array(g2)), AffineTransform, min_samples=3, residual_threshold=self.min_epsilon, max_trials=self.max_trials, stop_residuals_sum=self.min_inlier_ratio)
            
            self.stack[x+1,...] = warp(self.stack[x+1,...], AffineTransform(rotation=model.rotation, translation=model.translation), output_shape=self.stack[x+1].shape)

        self.stack = self.stack.astype(np.uint8)
        #return model
        #return k,x


    def run3(self):
        """ Cette fonction test des alternatives à SIFT et ORB. Ne fonctionne pas."""
        for x in xrange(len(self.stack)-1):
            print('Traitement image ' + str(x+1))
            im1,im2 = 255.*gaussian_filter(self.stack[x,...], sqrt(self.initial_sigma**2 - 0.25)), 255.*gaussian_filter(self.stack[x+1,...], sqrt(self.initial_sigma**2 - 0.25))
            im1,im2 = enhance_contrast(normaliser(im1), square(3)), enhance_contrast(normaliser(im2), square(3))
            im1, im2 = normaliser(im1), normaliser(im2)
            
            b = cv2.BRISK()
            #b.create("Feature2D.BRISK")
            
            k1,d1 = b.detectAndCompute(im1,None)
            k2,d2 = b.detectAndCompute(im2,None)
            
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = bf.match(d1,d2)
            
            g1,g2 = [],[]
            for i in matches:
                g1.append(k1[i.queryIdx].pt)
                g2.append(k2[i.trainIdx].pt)

            model, inliers = ransac((np.array(g1), np.array(g2)), AffineTransform, min_samples=3, residual_threshold=self.min_epsilon, max_trials=self.max_trials, stop_residuals_sum=self.min_inlier_ratio)
            
            self.stack[x+1,...] = warp(self.stack[x+1,...], AffineTransform(rotation=model.rotation, translation=model.translation), output_shape=self.stack[x+1].shape)

        self.stack = self.stack.astype(np.uint8)


    def run4(self):
        """ Cette fonction recadre les images grâce à SURF et RANSAC, fonctionne bien."""
        for x in xrange(len(self.stack)-1):
            print('Traitement image ' + str(x+1))
            im1,im2 = 255.*gaussian_filter(self.stack[x,...], sqrt(self.initial_sigma**2 - 0.25)), 255.*gaussian_filter(self.stack[x+1,...], sqrt(self.initial_sigma**2 - 0.25))
            im1,im2 = enhance_contrast(normaliser(im1), square(5)), enhance_contrast(normaliser(im2), square(5))
            im1, im2 = normaliser(im1), normaliser(im2)
            
            b = cv2.SURF()
            #b.create("Feature2D.BRISK")
            
            k1,d1 = b.detectAndCompute(im1,None)
            k2,d2 = b.detectAndCompute(im2,None)
            
            bf = cv2.BFMatcher()
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

            model, inliers = ransac((np.array(g1), np.array(g2)), AffineTransform, min_samples=3, residual_threshold=self.min_epsilon, max_trials=self.max_trials, stop_residuals_sum=self.min_inlier_ratio)
            
            self.stack[x+1,...] = warp(self.stack[x+1,...], AffineTransform(rotation=model.rotation, translation=model.translation), output_shape=self.stack[x+1].shape)

        self.stack = self.stack.astype(np.uint8)
            
        
        
        
        
        
        
        
# test

s = AlignStack('C:/Users/ad245339/Desktop/images/Dissolution Mesbah-Tocino 2011/AM39-FDA-suivi disso X5000')
s.run4()
s.cropage()
imsave('test.tif', s.stackAligne)
""" 
"""      
        
        
        
        
        
        
        
        
        
        
        
        
        
        