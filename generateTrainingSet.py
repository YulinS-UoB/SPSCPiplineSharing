# ***********************************
import skimage
import numpy as np
# ***********************************


class TrainingSetGenerator:

    def __init__(self, annotation_meta):
        self.labelMap = annotation_meta['label map']
        self.imgStack = np.load('{}.npy'.format(annotation_meta['image saving prefix']))
        self.maskStack = np.load('{}.npy'.format(annotation_meta['mask saving prefix']))
