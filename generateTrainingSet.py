# ***********************************
import skimage
import numpy as np
# ***********************************


class TrainingSetGenerator:

    def __init__(self, annotation_meta, vision_rad, vision_mode='TorchShine', vision_shape='circle', dim_oder='txyc'):
        self.labelMap = annotation_meta['label map']
        self.imgStack = np.load('{}.npy'.format(annotation_meta['image saving prefix']))
        self.maskStack = np.load('{}.npy'.format(annotation_meta['mask saving prefix']))
        self.imageDim = dim_oder
        # imageDim = ['txyc', 'xyct', 'xyc', 'xyt', 'txy']
        self.visionRad = vision_rad
        # Radius of vision field in pixel/sigma
        self.visionMod = vision_mode
        # [TorchShine: A vision field with small size and equal weight;
        #  MoonLight: A vision field with large size and decayed weight with gaussian distribution in distance]
        #  The vision_rad is the pixel rad in TorchShine and sigma in MoonLight
        self.visionShape = vision_shape
        # [circle, square]

    def arrangeDim(self):
        ndim = self.imgStack.ndim
        if self.imgStack.shape != self.maskStack.shape:
            print('Warning: Image shape and mask shape is not identical!')
        if not (self.imgStack.ndim == len(self.imageDim) and self.maskStack.ndim == len(self.imageDim)):
            print('Error: Either image or mask is not arranged in style of {}, check data dimension.'
                  .format(self.imageDim))
            pass
        else:
            if (self.imageDim in ['txyc', 'xyc']) and (self.imgStack.shape[-1] > 4 or self.maskStack.shape[-1] > 4):
                print('Warning: Found either image or mask has channel number over 4. '
                      'Are you sure the dim order is correct?')



