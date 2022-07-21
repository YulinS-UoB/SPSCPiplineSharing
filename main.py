# ***********************************
from readROIAnnotation import *
from generateTrainingSet import *
# ***********************************

annoFileList = ['RoiSetBackGrnd.zip', 'RoiSetForeGrnd.zip']
labelList = [1, 2]
stackPath = 'data/origin/StackOrigin.tif'
anno = AnnotationInterprator('data/origin/', annoFileList, labelList, stackPath,
                             'data/annotated_mask/mask_Batch-Row-Col', 'data/img_array/mainImgStack',
                             meta_save_prefix='data/annotation_meta')
anno.AIO_SelfComprehend()
annoMeta = anno.AIO_SelfSaving()

tsGen = TrainingSetGenerator(annoMeta, dataset_prefix='data/AnnotationDS', vision_rad=1, vision_mode='CandleLight',
                             dim_oder='txy', preprocess='versatile')
tsGen.arrangeDim()
tsGen.preProcess(save_by_iter=True, write_img=True)
