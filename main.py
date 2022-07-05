# ***********************************
from readROIAnnotation import *
# ***********************************
annoFileList = ['RoiSetBackGrnd.zip', 'RoiSetForeGrnd.zip']
labelList = [1, 2]
stackPath = 'data/origin/StackOrigin.tif'
anno = AnnotationInterprator('data/origin/', annoFileList, labelList, stackPath,
                             'data/annotated_mask/mask_Batch-Row-Col_')
anno.AIO_SelfComprehend()
