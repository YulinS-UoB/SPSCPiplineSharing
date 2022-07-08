# ***********************************
from skimage import morphology
from scipy import ndimage
import numpy as np
# ***********************************


class TrainingSetGenerator:

    def __init__(self, annotation_meta, vision_rad, vision_mode='TorchShine', vision_shape='circle',
                 preprocess='verstile', dim_oder='txyc', **kwargs):
        self.labelMap = annotation_meta['label map']
        self.imgStack = np.load('{}.npy'.format(annotation_meta['image saving prefix']))
        self.maskStack = np.load('{}.npy'.format(annotation_meta['mask saving prefix']))
        self.imageDim = dim_oder
        # imageDim = ['txyc', 'xyct', 'xyc', 'xyt', 'txy']
        self.visionRad = vision_rad
        # Radius of vision field in pixel/sigma
        self.visionMod = vision_mode
        # [TorchShine: A vision field with small size and equal weight;
        #  CandleLight: A vision field with large size and decayed weight with gaussian distribution in distance]
        #  The vision_rad is the pixel rad in TorchShine and sigma in MoonLight
        self.visionShape = vision_shape
        # [circle, square]
        protocol_temp = ['Gaussian', 'LoG', 'GEoG', 'DoG', 'STE', 'HoGE']
        sigma_choice = [0.2, 0.33, 0.6, 1, 1.8, 3, 5.4, 9, 16.2]
        self.processMode = {'versatile': [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6, 7, 8]],
                            'traditional': [[0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7, 8]]}
        self.preProcess = {'method': [protocol_temp[index1] for index1 in self.processMode[preprocess][0]],
                           'sigma': [sigma_choice[index2] for index2 in self.processMode[preprocess][1]]}
        '''
        Gaussian: Gaussian Blurr
        LoG: Laplacian of Gaussian
        GEoG: Gradient Energy of Gaussian
        DoG: Difference of Gaussian
        STE: Structure Tensor Eigenvalue
        HoGE: Hessian of Gaussian Eigenvalue
        '''
        self.generateMask = [None, None]
        # 0: Shape of the mask; 1: Value of the mask
        if self.visionMod == 'TorchShine':
            if self.visionShape == 'circle':
                self.generateMask[0] = morphology.disk(self.visionRad)
            elif self.visionShape == 'square':
                self.generateMask[0] = morphology.disk(self.visionRad * 2 + 1)
            self.generateMask[1] = self.generateMask[0]
        if self.visionMod == 'CandleLight':
            if self.visionShape == 'circle':
                self.generateMask[0] = morphology.disk(self.visionRad * 3)
            elif self.visionShape == 'square':
                self.generateMask[0] = morphology.disk(self.visionRad * 6 + 1)
            distance_mask = np.ones(self.generateMask[0].shape, dtype='float32')
            distance_mask[self.visionRad * 3, self.visionRad * 3] = 0
            distance_mask = ndimage.distance_transform_edt(
                np.exp(-np.square(distance_mask) / (2 * np.square(self.visionRad))) /
                (self.visionRad * np.sqrt(2 * np.pi))
            )
            self.generateMask[1] = self.generateMask[0] * distance_mask

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
