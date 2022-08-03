# ***********************************
from skimage import morphology, filters
from scipy import ndimage
import numpy as np
import datetime
import numba as nb
import os
import time
from scipy import linalg
import diplib as dip
# ***********************************

'''Below are Static Methods'''


@nb.jit()
def outerTensor(gr_x, gr_y):
    tensor = np.zeros((gr_x.shape[0], gr_x.shape[1], gr_x.shape[2], 4), dtype=np.double)
    for i in range(gr_x.shape[0]):
        for j in range(gr_x.shape[1]):
            for c in range(3):
                gr_v = [gr_x[i, j, c], gr_y[i, j, c]]
                tensor[i, j, c, :] = (np.outer(gr_v, gr_v)).flatten()
    return tensor


@nb.jit()
def calcuEigenVal(tensor):
    eigenval_tensor = np.zeros((tensor.shape[0], tensor.shape[1], tensor.shape[2], 2))
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            for c in range(tensor.shape[2]):
                eigenval_tensor[i, j, c, :] = linalg.eigvals(tensor[i, j, c, :].reshape((2, 2)))
    '''
    tensor_res = tensor.reshape((tensor.shape[0], tensor.shape[1], tensor.shape[2], 2, 2))
    eigenval_tensor = np.linalg.eigvals(tensor_res)
    '''
    return eigenval_tensor.astype('double')


def gaussianBlur(tensor, sigma):
    blurred_tensor = np.zeros(tensor.shape, dtype=np.double)
    for prdct in range(tensor.shape[3]):
        blurred_tensor[:, :, :, prdct] = filters.gaussian(tensor[:, :, :, prdct], sigma=sigma, channel_axis=2)
    return blurred_tensor


'''Above are Static Methods'''


class TrainingSetGenerator:

    def __init__(self, annotation_meta, vision_rad, vision_mode='TorchShine', vision_shape='circle',
                 preprocess='versatile', dim_oder='txyc', dataset_prefix='annotationDS', **kwargs):
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
        self.datasetName = '{}{}/'.format(dataset_prefix, datetime.datetime.now().strftime('%Y-%m-%d_%H_%M'))
        if not os.path.exists(self.datasetName):
            os.mkdir(self.datasetName)
        protocol_temp = ['Gaussian', 'LoG', 'GEoG', 'DoG', 'STE', 'HoGE']
        sigma_choice = [0.2, 0.33, 0.6, 1, 1.8, 3, 5.4, 9, 16.2]
        self.processMode = {'versatile': [[0, 1, 2, 3, 4, 5], [0, 2, 4, 6, 7, 8]],
                            'traditional': [[0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7, 8]],
                            'test': [[0, 1, 2, 3, 4, 5], [0]]}
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
        self.generateMask = [np.empty((0, 0)), np.empty((0, 0))]  # 0: Shape of the mask; 1: Value of the mask
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
        #  Convert image and mask to TXYC form
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

    def preProcess(self, save_by_iter=False, write_img=False):
        # pre-process the images according to the protocol, and save(or not) all the features (after processing) of the
        # labeled pixels as dataset
        # Has 3 sub functions:
        #   * imgOpt_at_sTimePoint (do the pre-processing)
        #   * iterRtrvFeatures (retrieve annotated pixels as multi-feature data point)
        #   * savePatchs (save the data points, and return the data points of 1 time point as a dictionary, either as
        #     path or array)
        trtmnt_num = 0
        for methodIdx in range(len(self.processProtocol['method'])):
            for sigmaIdx in range(len(self.processProtocol['sigma'])):
                if self.processProtocol['method'][methodIdx] in ['Gaussian', 'LoG', 'DoG', 'GEoG']:
                    trtmnt_num = trtmnt_num + 1
                else:
                    trtmnt_num = trtmnt_num + 2
        processed_dict = {'results': {}, 'meta': {}}
        for tIdx in range(self.imgStack.shape[0]):
            if self.maskStack[tIdx, :, :, :].sum(dtype='uint32') != 0:
                origin_img = self.imgStack[tIdx, :, :, :].astype(np.double)
                res_shape = (trtmnt_num, self.imgStack.shape[1], self.imgStack.shape[2], self.imgStack.shape[3])
                processed_stack = self.imgOpt_at_sTimePoint(origin_img, res_shape)

                for batchId in range(processed_stack.shape[0]):
                    for c in range(processed_stack.shape[3]):
                        min_val = processed_stack[batchId, :, :, c].min()
                        max_val = processed_stack[batchId, :, :, c].max()
                        processed_stack[batchId, :, :, c] = \
                            (processed_stack[batchId, :, :, c] - min_val) / (max_val - min_val + 1e-10) * 65535

                processed_stack = processed_stack.astype('uint16')

                # ↑ Normalization
                if write_img:
                    np.save('{}Processed_Stack_T{:05d}.npy'.format(self.datasetName, tIdx), processed_stack)
                if save_by_iter:
                    processed_dict['results']['{:05d}'.format(tIdx)] = self.savePatches(processed_stack, tIdx)
                else:
                    processed_dict['results']['{:05d}'.format(tIdx)] = self.savePatches(processed_stack, tIdx,
                                                                                        write_ds=False)
        return processed_dict

    def savePatches(self, res_stack, tidx, write_ds=True, max_px=10000):
        mask = self.maskStack[tidx, :, :, 0]
        label_loc = {}
        data_patch = []
        for label in self.labelMap.keys():
            label_loc[label] = np.array(np.where(mask == self.labelMap[label]))
        for label in label_loc.keys():
            batch_num = label_loc[label].shape[1]
            label_feature_instances = self.iterRtrvFeatures(batch_num, res_stack, label_loc[label])
            '''
            for istc in range(label_feature_instances.shape[0]):
                if write_ds:
                    data_patch.append({'label': label, 'feature': '{}{}/T{:05d}P{:08d}.npy'.format(self.datasetName,
                                                                                                   label, tidx, istc)})
                    if not os.path.exists('{}{}'.format(self.datasetName, label)):
                        os.mkdir('{}{}'.format(self.datasetName, label))
                    np.save('{}{}/T{:05d}P{:08d}.npy'.format(self.datasetName, label, tidx, istc),
                            label_feature_instances[istc])
                else:
                    data_patch.append({'label': label, 'feature': label_feature_instances[istc, :]})
            '''
            if write_ds:
                if label_feature_instances.shape[0] > max_px:
                    bar_num = (label_feature_instances.shape[0] // max_px) + 1
                    for bar in range(bar_num):
                        data_patch.append({'label': label, 'feature': '{}{}/T{:05d}B{:03d}.npy'.format(
                            self.datasetName, label, tidx, bar)})
                        if not os.path.exists('{}{}'.format(self.datasetName, label)):
                            os.mkdir('{}{}'.format(self.datasetName, label))
                        if (bar + 1) * 10000 < label_feature_instances.shape[0]:
                            np.save('{}{}/T{:05d}B{:03d}.npy'.format(self.datasetName, label, tidx, bar),
                                    label_feature_instances[bar * 10000:(bar + 1) * 10000, :])
                        else:
                            np.save('{}{}/T{:05d}B{:03d}.npy'.format(self.datasetName, label, tidx, bar),
                                    label_feature_instances[bar * 10000:label_feature_instances.shape[0], :])
            else:
                data_patch.append({'label': label, 'feature': label_feature_instances})
        return data_patch

    #  @nb.jit()
    def iterRtrvFeatures(self, batch_num, res_stack, label_loc):
        time0 = time.time()
        feature_array = np.zeros((batch_num, res_stack.shape[0] * res_stack.shape[3]))
        focus_rad = (self.generateMask[0].shape[0] - 1) // 2
        padded_feature = np.pad(res_stack, ((0, 0), (focus_rad, focus_rad), (focus_rad, focus_rad), (0, 0)),
                                mode='reflect')
        nd_filter = np.zeros((res_stack.shape[0], self.generateMask[0].shape[0], self.generateMask[0].shape[1],
                              res_stack.shape[3]))
        noticed_loc = np.where(self.generateMask[0] != 0)
        for t in range(res_stack.shape[0]):
            for c in range(res_stack.shape[3]):
                nd_filter[t, :, :, c] = self.generateMask[1]
        for loc in range(batch_num):
            x, y = label_loc[:, loc]
            x = x + focus_rad
            y = y + focus_rad
            feature_region = padded_feature[:, (x - focus_rad):(x + focus_rad + 1),
                                            (y - focus_rad):(y + focus_rad + 1), :]
            # noticed_feature = ((feature_region * nd_filter)[:, noticed_loc[0], noticed_loc[1], :]).flatten()
            noticed_feature = \
                ((feature_region * nd_filter)[:, noticed_loc[0], noticed_loc[1], :].mean(axis=(1, 2))).flatten()
            feature_array[loc, :] = noticed_feature
        print('{} secs for retrieving pixels in single res stack'.format(time.time() - time0))
        return feature_array

    def imgOpt_at_sTimePoint(self, image, shape):
        processed_stack = np.zeros(shape, dtype=np.double)
        origin_img = image
        # processed_stack shape: M-X-Y-C (preprocessing methods, x, y, channel)
        batch_pointer = 0  # Indicating which batch layer should the processed image be written into
        for methodIdx in range(len(self.processProtocol['method'])):
            method = self.processProtocol['method'][methodIdx]
            # Gaussian Blurr
            if method == 'Gaussian':
                time0 = time.time()
                for sigmaIdx in range(len(self.processProtocol['sigma'])):
                    sigma = self.processProtocol['sigma'][sigmaIdx]
                    processed_stack[batch_pointer, :, :, :] = \
                        filters.gaussian(origin_img, sigma=sigma, channel_axis=2)
                    batch_pointer = batch_pointer + 1
                print('{} secs for Gaussian blurr'.format(time.time() - time0))

            # Laplacian of Gaussian
            elif method == 'LoG':
                time0 = time.time()
                for sigmaIdx in range(len(self.processProtocol['sigma'])):
                    sigma = self.processProtocol['sigma'][sigmaIdx]
                    blurred = filters.gaussian(origin_img, sigma=sigma, channel_axis=2)
                    processed_stack[batch_pointer, :, :, :] = \
                        filters.laplace(blurred)
                    batch_pointer = batch_pointer + 1
                print('{} secs for Laplacian of Gaussian'.format(time.time() - time0))

            # Difference of Gaussian
            elif method == 'DoG':
                time0 = time.time()
                for sigmaIdx in range(len(self.processProtocol['sigma'])):
                    sigma = self.processProtocol['sigma'][sigmaIdx]
                    processed_stack[batch_pointer, :, :, :] = \
                        filters.difference_of_gaussians(origin_img, low_sigma=sigma, channel_axis=2)
                    batch_pointer = batch_pointer + 1
                print('{} secs for Difference of Gaussian'.format(time.time() - time0))

            # Gradient Magnitude of Gaussian
            elif method == 'GEoG':
                time0 = time.time()
                for sigmaIdx in range(len(self.processProtocol['sigma'])):
                    sigma = self.processProtocol['sigma'][sigmaIdx]
                    blurred = filters.gaussian(origin_img, sigma=sigma, channel_axis=2)
                    processed_stack[batch_pointer, :, :, :] = \
                        np.sqrt(np.square(np.gradient(blurred, axis=0)) +
                                np.square(np.gradient(blurred, axis=1)))
                    batch_pointer = batch_pointer + 1
                print('{} secs for Energy of Gaussian'.format(time.time() - time0))

            # Structure Tensor Eigenvalue
            elif method == 'STE':
                time0 = time.time()
                '''
                gradient_x = np.gradient(origin_img, axis=0)
                gradient_y = np.gradient(origin_img, axis=1)
                outer_tensor = outerTensor(gradient_x, gradient_y)
                '''
                for sigmaIdx in range(len(self.processProtocol['sigma'])):
                    for c in range(origin_img.shape[2]):
                        sigma = self.processProtocol['sigma'][sigmaIdx]
                        blurred_tensor = dip.StructureTensor(origin_img[:, :, c], tensorSigmas=sigma)
                        eigenval, eigenvec = dip.EigenDecomposition(blurred_tensor)
                        '''
                        blurred_tensor = gaussianBlur(outer_tensor, sigma)
                        eigen_res = calcuEigenVal(blurred_tensor)
                        eigen_res = np.moveaxis(eigen_res, 3, 0)
                        '''
                        processed_stack[batch_pointer, :, :, c] = np.asarray(eigenval(0))
                        processed_stack[batch_pointer + 1, :, :, c] = np.asarray(eigenval(1))
                    batch_pointer = batch_pointer + 2
                print('{} secs for Structure Tensor Eigenvalue'.format(time.time() - time0))

            # Hessian of Gaussian Eigenvalue
            elif method == 'HoGE':
                time0 = time.time()
                for sigmaIdx in range(len(self.processProtocol['sigma'])):
                    sigma = self.processProtocol['sigma'][sigmaIdx]
                    blurred_tensor = filters.gaussian(origin_img, sigma=sigma, channel_axis=2)
                    '''
                    outer_tensor = np.zeros((origin_img.shape[0], origin_img.shape[1],
                                             origin_img.shape[2], 4))
                    '''
                    for c in range(origin_img.shape[2]):
                        '''
                        outer_tensor[:, :, c, 0] = \
                            ndimage.gaussian_filter(origin_img[:, :, c], sigma=sigma, order=(2, 0))
                        outer_tensor[:, :, c, 3] = \
                            ndimage.gaussian_filter(origin_img[:, :, c], sigma=sigma, order=(0, 2))
                        even_driv = ndimage.gaussian_filter(origin_img[:, :, c], sigma=sigma, order=(1, 1))
                        outer_tensor[:, :, c, 1] = even_driv
                        outer_tensor[:, :, c, 2] = even_driv
                    eigen_res = calcuEigenVal(outer_tensor)
                    eigen_res = np.moveaxis(eigen_res, 3, 0)
                    '''
                        hessian_img = dip.Hessian(blurred_tensor[:, :, c], sigmas=sigma)
                        eigenval, eigenvec = dip.EigenDecomposition(hessian_img)
                        processed_stack[batch_pointer, :, :, c] = np.asarray(eigenval(0))
                        processed_stack[batch_pointer + 1, :, :, c] = np.asarray(eigenval(1))
                    batch_pointer = batch_pointer + 2
                print('{} secs for Hessian of Gaussian Eigenvalue'.format(time.time() - time0))
        return processed_stack
