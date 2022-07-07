# ***********************************
import numpy as np
from roifile import ImagejRoi
import skimage.io as skio
from skimage.draw import polygon
import json as js
# ************************************


class AnnotationInterprator:

    def __init__(self, annotation_dir, annotation_list, annotation_label, image_stack_path,
                 mask_save_prefix, img_save_prefix, meta_save_prefix=None, image_type='uint16'):
        self.annotationMap = {}
        # Dict mapping annotation label and corresponding ROI file path
        self.labelMap = {}
        # Dict mapping annotation label and corresponding mask region value
        self.annotationROI = {}
        # Dict mapping annotation label and corresponding ImagJ ROI object
        self.imagePath = image_stack_path
        self.imgType = image_type
        self.maskPrefix = mask_save_prefix
        self.imgPrefix = img_save_prefix
        self.metaPrefix = meta_save_prefix
        self.IMGStack = None
        self.IMGShape = None
        self.IMGMask = None

        for labelIdx in range(len(annotation_label)):
            if type(annotation_label[labelIdx]) == int:
                text_label = '{:03d}'.format(annotation_label[labelIdx])
                self.labelMap[text_label] = annotation_label[labelIdx]
            elif not type(annotation_label[labelIdx]) == str:
                text_label = str(annotation_label[labelIdx])
                self.labelMap[text_label] = labelIdx + 1
            else:
                text_label = annotation_label[labelIdx]
                self.labelMap[text_label] = labelIdx + 1
            self.annotationMap[text_label] = annotation_dir + annotation_list[labelIdx]

    def getAnnoInfo(self):
        self.IMGStack = skio.imread(self.imagePath, plugin="tifffile").astype(self.imgType)
        self.IMGShape = self.IMGStack.shape
        self.IMGMask = np.zeros(self.IMGShape, dtype='uint8')

        for label in self.annotationMap.keys():
            self.annotationROI[label] = ImagejRoi.fromfile(self.annotationMap[label])

    def generateMask(self):
        for tPos in range(self.IMGMask.shape[0]):
            for label in self.annotationROI.keys():
                for roiIdx in range(len(self.annotationROI[label])):
                    roi = self.annotationROI[label][roiIdx]
                    if roi.position == (tPos + 1):
                        vtx_list = roi.coordinates(multi=True)
                        for regionIdx in range(len(vtx_list)):
                            region = vtx_list[regionIdx]
                            rpos = region[:, 1]
                            cpos = region[:, 0]
                            rr, cc = polygon(rpos, cpos)
                            self.IMGMask[tPos, rr, cc] = self.labelMap[label]

    def saveMask(self):
        np.save('{}.npy'.format(self.maskPrefix), self.IMGMask)

    def saveImg(self):
        np.save('{}.npy'.format(self.imgPrefix), self.IMGStack)

    def saveMeta(self, save=True, return_meta=True):
        content = {'raw image path': self.imagePath, 'annotation file path': self.annotationMap,
                   'label map': self.labelMap, 'image set shape': self.IMGStack.shape,
                   'image saving prefix': self.imgPrefix, 'mask saving prefix': self.maskPrefix,
                   'meta saving prefix': self.metaPrefix}
        if save:
            with open('{}.json'.format(self.metaPrefix), 'w') as jf:
                js.dump(content, jf)
        if return_meta:
            return content

    def AIO_SelfComprehend(self, return_roi=False):
        self.getAnnoInfo()
        self.generateMask()
        if return_roi:
            return self.annotationROI

    def AIO_SelfSaving(self, return_meta=True):
        self.saveMask()
        self.saveImg()
        if return_meta:
            meta = self.saveMeta(save=self.metaPrefix)
            return meta
        else:
            self.saveMeta(save=self.metaPrefix, return_meta=False)
