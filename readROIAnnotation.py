# ***********************************
import numpy as np
from roifile import ImagejRoi
import skimage.io as skio
from skimage.draw import polygon
# ************************************


class AnnotationInterprator:

    def __init__(self, annotation_dir, annotation_list, annotation_label, image_stack_path,
                 mask_save_prefix, img_save_prefix, image_type='uint16'):

        self.annotationMap = {}
        self.labelMap = {}
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
        # 注释编号与注释文件路径相对应的Dict

        self.annotationROI = {}
        # 注释编号与ImageJ ROI对象相对应的Dict

        self.imagePath = image_stack_path
        self.imgType = image_type
        self.maskPrefix = mask_save_prefix
        self.imgPrefix = img_save_prefix
        self.IMGStack = None
        self.IMGShape = None
        self.IMGMask = {}

    def getAnnoInfo(self):
        self.IMGStack = skio.imread(self.imagePath, plugin="tifffile").astype(self.imgType)
        self.IMGShape = self.IMGStack.shape

        for label in self.annotationMap.keys():
            self.annotationROI[label] = ImagejRoi.fromfile(self.annotationMap[label])
            self.IMGMask[label] = np.zeros(self.IMGShape, dtype='uint8')

    def generateMask(self):
        for label in self.annotationROI.keys():
            for tPos in range(self.IMGMask[label].shape[0]):
                for roiIdx in range(len(self.annotationROI[label])):
                    roi = self.annotationROI[label][roiIdx]
                    if roi.position == (tPos + 1):
                        vtx_list = roi.coordinates(multi=True)
                        for regionIdx in range(len(vtx_list)):
                            region = vtx_list[regionIdx]
                            rpos = region[:, 1]
                            cpos = region[:, 0]
                            rr, cc = polygon(rpos, cpos)
                            self.IMGMask[label][tPos, rr, cc] = self.labelMap[label]

    def saveMask(self):
        for label in self.IMGMask.keys():
            np.save('{}{}.npy'.format(self.maskPrefix, label), self.IMGMask[label])

    def saveImg(self):
        np.save('{}.npy'.format(self.imgPrefix), self.IMGStack)

    def AIO_SelfComprehend(self):
        self.getAnnoInfo()
        self.generateMask()

    def AIO_SelfSaving(self):
        self.saveMask()
        self.saveImg()

