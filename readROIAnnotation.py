# ***********************************
import cv2 as cv
import numpy as np
import os
from roifile import ImagejRoi
import skimage.io as skio
from skimage.draw import polygon
import matplotlib.pyplot as plt
# ************************************


class AnnotationInterprator:

    def __init__(self, annotation_list, image_type='uint16'):
        self.pathFGAnno = annotation_list[0]
        self.pathBGAnno = annotation_list[1]
        self.imageStack = annotation_list[2]
        self.imgType = image_type
    def 
