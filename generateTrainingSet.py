# ***********************************
import skimage
import numpy as np
# ***********************************


class TrainingSetGenerator:

    def __init__(self, annotation_meta):
        self.labelMap = annotation_meta['label map']
        
