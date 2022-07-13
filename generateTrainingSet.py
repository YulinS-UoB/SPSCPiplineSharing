# ***********************************
from skimage import morphology, filters
from scipy import ndimage
import numpy as np
import datetime
# ***********************************


class TrainingSetGenerator:

    def __init__(self, annotation_meta, vision_rad, vision_mode='TorchShine', vision_shape='circle',
                 preprocess='verstile', dim_oder='txyc', dataset_prefix='annotationDS', **kwargs):
        self.labelMap = annotation_meta['label map']
        self.imgStack = np.load('{}.npy'.format(annotation_meta['image saving prefix']))
        self.maskStack = np.load('{}.npy'.format(annotation_meta['mask saving prefix']))
        self.imageDim = dim_oder
        # imageDim = ['txyc', 'xyct', 'xyc', 'xyt', 'txy']
        # Note that: X, Y in the script all actually refer to row and column, respectively. They are NOT real X and Y!!
        self.visionRad = vision_rad
        # Radius of vision field in pixel/sigma
        self.visionMod = vision_mode
        # [TorchShine: A vision field with small size and equal weight;
        #  CandleLight: A vision field with large size and decayed weight with gaussian distribution in distance]
        #  The vision_rad is the pixel rad in TorchShine and sigma in MoonLight
        self.visionShape = vision_shape
        # [circle, square]
        self.datasetName = '{}{}.h5'.format(dataset_prefix, datetime.datetime.now().strftime('%Y-%m-%d_%H_%M'))
        protocol_temp = ['Gaussian', 'LoG', 'GEoG', 'DoG', 'STE', 'HoGE']
        sigma_choice = [0.2, 0.33, 0.6, 1, 1.8, 3, 5.4, 9, 16.2]
        self.processMode = {'versatile': [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5, 6, 7, 8]],
                            'traditional': [[0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7, 8]]}
        self.processProtocol = {'method': [protocol_temp[index1] for index1 in self.processMode[preprocess][0]],
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
                self.generateMask[0] = morphology.square(self.visionRad * 2 + 1)
            self.generateMask[1] = self.generateMask[0]
        elif self.visionMod == 'CandleLight':
            if self.visionShape == 'circle':
                self.generateMask[0] = morphology.disk(self.visionRad * 3)
            elif self.visionShape == 'square':
                self.generateMask[0] = morphology.square(self.visionRad * 6 + 1)
            distance_mask = np.ones(self.generateMask[0].shape, dtype='float32')
            distance_mask[self.visionRad * 3, self.visionRad * 3] = 0
            distance_mask = ndimage.distance_transform_edt(
                np.exp(-np.square(distance_mask) / (2 * np.square(self.visionRad))) /
                (self.visionRad * np.sqrt(2 * np.pi))
            )
            self.generateMask[1] = self.generateMask[0] * distance_mask

    def arrangeDim(self):

        ndim = self.imgStack.ndim
        sc_mask = False  # If Mask Stack got channel dim reduced
        if self.imgStack.shape != self.maskStack.shape:
            if ndim == self.maskStack.ndim + 1:
                proposed_mask_shape = [self.imgStack.shape[dim] for dim in range(len(ndim))
                                       if dim != self.imageDim.find('c')]
                if self.maskStack.shape == proposed_mask_shape:
                    print('Omitting the channel dim in mask stack...')
                    sc_mask = True
                else:
                    print('Error: Image shape and mask shape is not identical!')
                    return 1
            else:
                print('Error: Mask dim is strange, check data dimension')
                return 2
        else:
            pass

        if not ndim == len(self.imageDim):
            print('Error: Either image or mask does not have dim number as input {}, check data dimension.'
                  .format(self.imageDim))
            return 3
        else:
            pass

        if (self.imageDim in ['txyc', 'xyc']) and (self.imgStack.shape[-1] > 3 or self.maskStack.shape[-1] > 3):
            print('Warning: Found either image or mask has channel number over 3. '
                  'Are you sure the dim order is correct?')
        else:
            pass

        # imageDim = ['txyc', 'xyct', 'xyc', 'xyt', 'txy']

        if self.imageDim == 'txyc':
            if not sc_mask:
                pass
            else:
                self.maskStack = np.expand_dims(self.maskStack, axis=3)
        elif self.imageDim == 'txy':
            self.imgStack = np.expand_dims(self.imgStack, axis=3)
            self.maskStack = np.expand_dims(self.maskStack, axis=3)
        elif self.imageDim == 'xyt':
            self.imgStack = np.expand_dims(self.imgStack, axis=3)
            self.maskStack = np.expand_dims(self.maskStack, axis=3)
            self.imgStack = np.moveaxis(self.imgStack, 2, 0)
            self.maskStack = np.moveaxis(self.maskStack, 2, 0)
        elif self.imageDim == 'xyc':
            if not sc_mask:
                self.imgStack = np.expand_dims(self.imgStack, axis=0)
                self.maskStack = np.expand_dims(self.maskStack, axis=0)
            else:
                self.imgStack = np.expand_dims(self.imgStack, axis=0)
                self.maskStack = np.expand_dims(self.maskStack, axis=(0, 3))
        elif self.imageDim == 'xyct':
            if not sc_mask:
                self.imgStack = np.moveaxis(self.imgStack, 3, 0)
                self.maskStack = np.moveaxis(self.maskStack, 3, 0)
            else:
                self.imgStack = np.moveaxis(self.imgStack, 3, 0)
                self.maskStack = np.expand_dims(self.maskStack, axis=2)
                self.maskStack = np.moveaxis(self.maskStack, 3, 0)
        # Now both image stack and mask stack become txyc style, 4 dimensions

    def preProcess(self):
        for tIdx in range(self.imgStack.shape[0]):
            if self.maskStack[tIdx, :, :, :].sum(dtype='uint32') != 0:
                trtmnt_num = len(self.processProtocol['method']) * len(self.processProtocol['sigma'])
                processed_stack = np.zeros((trtmnt_num, self.imgStack.shape[1], self.imgStack.shape[2],
                                           self.imgStack.shape[3]), dtype=self.imgStack.dtype)
                # processed_stack shape: M-X-Y-C (preprocessing methods, x, y, channel)
                for methodIdx in range(len(self.processProtocol['method'])):
                    method = self.processProtocol['method'][methodIdx]
                    # Gaussian Blurr
                    if method == 'Gaussian':
                        for sigmaIdx in range(len(self.processProtocol['sigma'])):
                            sigma = self.processProtocol['sigma'][sigmaIdx]
                            processed_stack[methodIdx * len(self.processProtocol['sigma']) + sigmaIdx, :, :, :] = \
                                filters.gaussian(self.imgStack.shape[tIdx, :, :, :], sigma=sigma, channel_axis=2)
                    # Laplacian of Gaussian
                    elif method == 'LoG':
                        for sigmaIdx in range(len(self.processProtocol['sigma'])):
                            sigma = self.processProtocol['sigma'][sigmaIdx]
                            blurred = filters.gaussian(self.imgStack.shape[tIdx, :, :, :], sigma=sigma, channel_axis=2)
                            processed_stack[methodIdx * len(self.processProtocol['sigma']) + sigmaIdx, :, :, :] = \
                                filters.laplace(blurred)
                    # Difference of Gaussian
                    elif method == 'DoG':
                        for sigmaIdx in range(len(self.processProtocol['sigma'])):
                            sigma = self.processProtocol['sigma'][sigmaIdx]
                            processed_stack[methodIdx * len(self.processProtocol['sigma']) + sigmaIdx, :, :, :] = \
                                filters.difference_of_gaussians(self.imgStack.shape[tIdx, :, :, :], low_sigma=sigma,
                                                                channel_axis=2)
                    # Gradient Magnitude (from Sobel) of Gaussian
                    elif method == 'GEoG':
                        for sigmaIdx in range(len(self.processProtocol['sigma'])):
                            sigma = self.processProtocol['sigma'][sigmaIdx]
                            blurred = filters.gaussian(self.imgStack.shape[tIdx, :, :, :], sigma=sigma, channel_axis=2)
                            processed_stack[methodIdx * len(self.processProtocol['sigma']) + sigmaIdx, :, :, :] = \
                                np.sqrt(np.square(filters.sobel(blurred, axis=0)) +
                                        np.square(filters.sobel(blurred, axis=1)))
                    # Structure Tensor Eigenvalue
                    elif method == 'STE':
                        

