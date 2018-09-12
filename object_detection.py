import numpy as np
import pandas as pd
from enum import Enum
from collections import Counter
import sys
import time
    

class CoordinatesType(Enum):
    """
    Class representing if the coordinates are relative to the
    image size or are absolute values.
    """
    Relative = 1
    Absolute = 2
    
class BBType(Enum):
    """
    Class representing if the bounding box is groundtruth or not.
    """
    GroundTruth = 1
    Detected = 2

class BBFormat(Enum):
    """
    Class representing the format of a bounding box.
    It can be (X,Y,width,height) => XYWH 
    or (X1,Y1,X2,Y2) => XYX2Y2
    """
    XYWH = 1
    XYX2Y2 = 2

# size => (width, height) of the image
# box => (X1, X2, Y1, Y2) of the bounding box
def convertToRelativeValues(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    cx = (box[1] + box[0])/2.0 
    cy = (box[3] + box[2])/2.0 
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = cx*dw
    y = cy*dh
    w = w*dw
    h = h*dh
    # x,y => (bounding_box_center)/width_of_the_image
    # w => bounding_box_width / width_of_the_image
    # h => bounding_box_height / height_of_the_image
    return (x,y,w,h)
        
# size => (width, height) of the image
# box => (centerX, centerY, w, h) of the bounding box relative to the image
def convertToAbsoluteValues(size, box):
    w_box = round(size[0] * box[2])
    h_box = round(size[1] * box[3])
    
    xIn = round(((2*float(box[0]) - float(box[2]))*size[0]/2))
    yIn = round(((2*float(box[1]) - float(box[3]))*size[1]/2))

    xEnd = xIn + round(float(box[2])*size[0])
    yEnd = yIn + round(float(box[3])*size[1])

    if xIn < 0:
        xIn = 0
    if yIn < 0:
        yIn = 0
    if xEnd >= size[0]:
        xEnd = size[0]-1
    if yEnd >= size[1]:
        yEnd = size[1]-1
    return (xIn,yIn,xEnd,yEnd)

###################################################################################

class BoundingBox:
    def __init__(self, imageName, classId, x, y, w, h, typeCoordinates = CoordinatesType.Absolute, imgSize = None, bbType=BBType.GroundTruth, classConfidence=None, format=BBFormat.XYWH):
        """Constructor.
        Args:
            imageName: String representing the image name.
            classId: String value representing class id.
            x: Float value representing the X upper-left coordinate of the bounding box.
            y: Float value representing the Y upper-left coordinate of the bounding box.
            w: Float value representing the width bounding box.
            h: Float value representing the height bounding box.
            typeCoordinates: (optional) Enum (Relative or Absolute) represents if the bounding box coordinates (x,y,w,h) are absolute or relative to size of the image. Default: 'Absolute'.
            imgSize: (optional) 2D vector (width, height)=>(int, int) represents the size of the image of the bounding box. If typeCoordinates is 'Relative', imgSize is required.
            bbType: (optional) Enum (Groundtruth or Detection) identifies if the bounding box represents a ground truth or a detection. If it is a detection, the classConfidence has to be informed.
            classConfidence: (optional) Float value representing the confidence of the detected class. If detectionType is Detection, classConfidence needs to be informed.
            format: (optional) Enum (BBFormat.XYWH or BBFormat.XYX2Y2) indicating the format of the coordinates of the bounding boxes. BBFormat.XYWH: <left> <top> <width> <height>  BBFormat.XYX2Y2: <left> <top> <right> <bottom>.
        """
        self._imageName = imageName
        self._typeCoordinates = typeCoordinates
        if typeCoordinates == CoordinatesType.Relative and imgSize == None:
            raise IOError('Parameter \'imgSize\' is required. It is necessary to inform the image size.')
        if bbType == BBType.Detected and classConfidence == None:
            raise IOError('For bbType=\'Detection\', it is necessary to inform the classConfidence value.')
        # if classConfidence != None and (classConfidence < 0 or classConfidence > 1):
            # raise IOError('classConfidence value must be a real value between 0 and 1. Value: %f' % classConfidence)

        self._classConfidence = classConfidence
        self._bbType = bbType
        self._classId = classId
        self._format = format

        # If relative coordinates, convert to absolute values
        if (typeCoordinates == CoordinatesType.Relative):
            (self._x,self._y,self._w,self._h) = convertToAbsoluteValues(imgSize, (x,y,w,h))
            self._width_img = imgSize[0]
            self._height_img =  imgSize[1]
            if format==BBFormat.XYWH:
                self._x2 = self._w
                self._y2 = self._h
                self._w = self._x2-self._x
                self._h = self._y2-self._y
            else:
                # Needed to implement
                raise IOError('To implement')
        else:
            self._x = x
            self._y = y
            if format==BBFormat.XYWH:
                self._w = w
                self._h = h
                self._x2 = self._x+self._w
                self._y2 = self._y+self._h
            else:
                self._x2 = w
                self._y2 = h
                self._w = self._x2-self._x
                self._h = self._y2+self._y
        if imgSize == None:
            self._width_img = None
            self._height_img =  None
        else:
            self._width_img = imgSize[0]
            self._height_img =  imgSize[1]

    def getAbsoluteBoundingBox(self, format=BBFormat.XYWH):
        if format == BBFormat.XYWH:
            return (self._x,self._y,self._w,self._h)
        elif format == BBFormat.XYX2Y2:
            return (self._x,self._y,self._x2,self._y2)

    def getRelativeBoundingBox(self, imgSize=None):
        if imgSize==None and self._width_img==None and self._height_img==None:
            raise IOError('Parameter \'imgSize\' is required. It is necessary to inform the image size.')
        if imgSize==None:
            return convertToRelativeValues((imgSize[0], imgSize[1]), (self._x,self._y,self._w,self._h))
        else:
            return convertToRelativeValues((self._width_img, self._height_img), (self._x,self._y,self._w,self._h))
    
    def getImageName(self):
        return self._imageName

    def getConfidence(self):
        return self._classConfidence
    
    def getFormat(self):
        return self._format

    def getClassId(self):
        return self._classId

    def getImageSize(self):
        return (self._width_img, self._height_img)

    def getCoordinatesType(self):
        return self._typeCoordinates
    
    def getBBType(self):
        return self._bbType
        
    @staticmethod
    def compare(det1, det2):
        det1BB = det1.getAbsoluteBoundingBox()
        det1ImgSize = det1.getImageSize()
        det2BB = det2.getAbsoluteBoundingBox()
        det2ImgSize = det2.getImageSize()
        
        if det1.getClassId() == det2.getClassId() and \
           det1.classConfidence == det2.classConfidenc() and \
           det1BB[0] == det2BB[0] and \
           det1BB[1] == det2BB[1] and \
           det1BB[2] == det2BB[2] and \
           det1BB[3] == det2BB[3] and \
           det1ImgSize[0] == det1ImgSize[0] and \
           det2ImgSize[1] == det2ImgSize[1]:
           return True
        return False     
    
    @staticmethod
    def clone(boundingBox):
        absBB = boundingBox.getAbsoluteBoundingBox(format=BBFormat.XYWH)
        # return (self._x,self._y,self._x2,self._y2)
        newBoundingBox = BoundingBox(boundingBox.getImageName(), boundingBox.getClassId(), \
                                    absBB[0], absBB[1], absBB[2], absBB[3], \
                                    typeCoordinates = boundingBox.getCoordinatesType(), \
                                    imgSize = boundingBox.getImageSize(), \
                                    bbType = boundingBox.getBBType(), \
                                    classConfidence = boundingBox.getConfidence(), \
                                    format = boundingBox.getFormat())
        return newBoundingBox

#######################################################################################


class BoundingBoxes:

    def __init__(self):
        self._boundingBoxes = []

    def addBoundingBox(self, bb):
        self._boundingBoxes.append(bb)

    def removeBoundingBox(self, _boundingBox):
        for d in self._boundingBoxes:
            if BoundingBox.compare(d,_boundingBox):
                del self._boundingBoxes[d]
                return
    
    def removeAllBoundingBoxes(self):
        self._boundingBoxes = []
    
    def getBoundingBoxes(self):
        return self._boundingBoxes

    def getBoundingBoxByClass(self, classId):
        boundingBoxes = []
        for d in self._boundingBoxes:
            if d.getClassId() == classId: # get only specified bounding box type
                boundingBoxes.append(d)
        return boundingBoxes

    def getClasses(self):
        classes = []
        for d in self._boundingBoxes:
            c = d.getClassId()
            if c not in classes:
                classes.append(c)
        return classes

    def getBoundingBoxesByType(self, bbType):
        # get only specified bb type
        return [d for d in self._boundingBoxes if d.getBBType() == bbType] 

    def getBoundingBoxesByImageName(self, imageName):
        # get only specified bb type
        return [d for d in self._boundingBoxes if d.getImageName() == imageName] 

    def count(self, bbType=None):
        if bbType == None: # Return all bounding boxes
            return len(self._boundingBoxes)
        count = 0
        for d in self._boundingBoxes:
            if d.getBBType() == bbType: # get only specified bb type
                count += 1
        return count
    
    def clone(self):
        newBoundingBoxes = BoundingBoxes()
        for d in self._boundingBoxes:
            det = BoundingBox.clone(d)
            newBoundingBoxes.addBoundingBox(det)
        return newBoundingBoxes







###########################################################################################
#                                                                                         #
# Evaluator class: Implements the most popular metrics for object detection               #
#                                       #
###########################################################################################


class Evaluator:
    
    def GetPascalVOCMetrics(self, boundingboxes, IOUThreshold=0.5):
        """Get the metrics used by the VOC Pascal 2012 challenge.
        Get
        Args:
            boundingboxes: Object of the class BoundingBoxes representing ground truth and detected bounding boxes;
            IOUThreshold: IOU threshold indicating which detections will be considered TP or FP (default value = 0.5).
        Returns:
            A list of dictionaries. Each dictionary contains information and metrics of each class. 
            The keys of each dictionary are: 
            dict['class']: class representing the current dictionary; 
            dict['precision']: array with the precision values;
            dict['recall']: array with the recall values;
            dict['AP']: average precision;
            dict['interpolated precision']: interpolated precision values;
            dict['interpolated recall']: interpolated recall values;
            dict['total positives']: total number of ground truth positives;
            dict['total TP']: total number of True Positive detections;
            dict['total FP']: total number of False Negative detections;
        """
        ret = [] # list containing metrics (precision, recall, average precision) of each class
        # List with all ground truths (Ex: [imageName, class, confidence=1, (bb coordinates XYX2Y2)])
        groundTruths = [] 
        # List with all detections (Ex: [imageName, class, confidence, (bb coordinates XYX2Y2)])
        detections = []
        # Get all classes
        classes = []
        # Loop through all bounding boxes and separate them into GTs and detections
        for bb in boundingboxes.getBoundingBoxes():
            # [imageName, class, confidence, (bb coordinates XYX2Y2)]
            if bb.getBBType() == BBType.GroundTruth:
                groundTruths.append([bb.getImageName(), bb.getClassId(), 1, bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2)])
            else:
                detections.append([bb.getImageName(), bb.getClassId(), bb.getConfidence(), bb.getAbsoluteBoundingBox(BBFormat.XYX2Y2)])
            # get class
            if bb.getClassId() not in classes:
                classes.append(bb.getClassId())
        classes = sorted(classes)
        ## Precision x Recall is obtained individually by each class
        # Loop through by classes
        for c in classes:
            # Get only detection of class c
            dects = []
            [dects.append(d) for d in detections if d[1] == c]
            # Get only ground truths of class c
            gts = []
            [gts.append(g) for g in groundTruths if g[1] == c]
            npos = len(gts)
            # sort detections by decreasing confidence
            dects = sorted(dects, key=lambda conf: conf[2], reverse=True)
            TP = np.zeros(len(dects))
            FP = np.zeros(len(dects))
            # create dictionary with amount of gts for each image
            det = Counter([cc[0] for cc in gts])
            for key,val in det.items():
                det[key] = np.zeros(val)
            # print("Evaluating class: %s (%d detections)" % (str(c), len(dects)))
            # Loop through detections
            for d in range(len(dects)):
                # print('dect %s => %s' % (dects[d][0], dects[d][3],))
                # Find ground truth image
                gt = [gt for gt in gts if gt[0] == dects[d][0]]
                iouMax = sys.float_info.min
                for j in range(len(gt)):
                    # print('Ground truth gt => %s' % (gt[j][3],))
                    iou = Evaluator.iou(dects[d][3], gt[j][3])
                    if iou>iouMax:
                        iouMax=iou
                        jmax=j
                # Assign detection as true positive/don't care/false positive
                if iouMax>=IOUThreshold:
                    if det[dects[d][0]][jmax] == 0:
                        TP[d]=1  # count as true positive
                        # print("TP")
                    det[dects[d][0]][jmax]=1 # flag as already 'seen'
                # - A detected "cat" is overlaped with a GT "cat" with IOU >= IOUThreshold.
                else:
                    FP[d]=1 # count as false positive
                    # print("FP")
            # compute precision, recall and average precision
            acc_FP=np.cumsum(FP)
            acc_TP=np.cumsum(TP)
            rec=acc_TP/npos
            prec=np.divide(acc_TP,(acc_FP+acc_TP))
            # print(len(rec), len(prec))
            [ap, mpre, mrec, ii] = Evaluator.CalculateAveragePrecision(rec, prec)
            # add class result in the dictionary to be returned
            r = {
                'class': c,
                'precision' : prec,
                'recall': rec,
                'AP': ap,
                'interpolated precision': mpre,
                'interpolated recall': mrec,
                'total positives': npos,
                'total TP': np.sum(TP),
                'total FP': np.sum(FP)                
                }
            ret.append(r)
        return ret


    @staticmethod
    def CalculateAveragePrecision(rec, prec):
        mrec = []
        mrec.append(0)
        [mrec.append(e) for e in rec]
        mrec.append(1)
        mpre = []
        mpre.append(0)
        [mpre.append(e) for e in prec]
        mpre.append(0)

        ii = []
        for i,j in zip(range(len(mpre)-1, 0, -1),range(len(mrec)-1)):
            mpre[i-1]=max(mpre[i-1],mpre[i])
            if mrec[j]!=mrec[j+1]:
                ii.append(j+1)
        
        
        ap = 0
        for i in ii:
            ap = ap + np.sum((mrec[i]-mrec[i-1])*mpre[i])
        
        return [ap, mpre[0:len(mpre)-1], mrec[0:len(mpre)-1], ii]

    # For each detections, calculate IOU with reference
    @staticmethod
    def _getAllIOUs(reference, detections):
        ret = []
        bbReference = reference.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
        # img = np.zeros((200,200,3), np.uint8)
        for d in detections:
            bb = d.getAbsoluteBoundingBox(BBFormat.XYX2Y2)
            iou = Evaluator.iou(bbReference,bb)
            # Show blank image with the bounding boxes
            # img = add_bb_into_image(img, d, color=(255,0,0), thickness=2, label=None)
            # img = add_bb_into_image(img, reference, color=(0,255,0), thickness=2, label=None)
            ret.append((iou,reference,d)) # iou, reference, detection
        # cv2.imshow("comparing",img)
        # cv2.waitKey(0)
        # cv2.destroyWindow("comparing")
        return sorted(ret, key=lambda i: i[0], reverse=True)# sort by iou (from highest to lowest)

    @staticmethod
    def iou(boxA, boxB):
        # if boxes dont intersect
        if Evaluator._boxesIntersect(boxA, boxB) == False:
            return 0
        interArea = Evaluator._getIntersectionArea(boxA,boxB)
        union = Evaluator._getUnionAreas(boxA,boxB,interArea=interArea)
        # intersection over union
        iou = interArea / union
        assert iou >= 0
        return iou

    # boxA = (Ax1,Ay1,Ax2,Ay2)
    # boxB = (Bx1,By1,Bx2,By2)
    @staticmethod
    def _boxesIntersect(boxA, boxB):
        if boxA[0] > boxB[2]: 
            return False # boxA is right of boxB
        if boxB[0] > boxA[2]:
            return False # boxA is left of boxB
        if boxA[3] < boxB[1]:
            return False # boxA is above boxB
        if boxA[1] > boxB[3]:
            return False # boxA is below boxB
        return True
    
    @staticmethod
    def _getIntersectionArea(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # intersection area
        return (xB - xA + 1) * (yB - yA + 1)
    
    @staticmethod
    def _getUnionAreas(boxA, boxB, interArea=None):
        area_A = Evaluator._getArea(boxA)
        area_B = Evaluator._getArea(boxB)
        if interArea == None:
            interArea = Evaluator._getIntersectionArea(boxA, boxB)
        return float(area_A + area_B - interArea)

    @staticmethod
    def _getArea(box):
        return (box[2] - box[0] + 1) * (box[3] - box[1] + 1)


###########################################################################################
#                                                                                         #
# This sample shows how to evaluate object detections applying the following metrics:     #
#  * Precision x Recall curve       ---->       used by VOC PASCAL 2012)                  #
#  * Average Precision (AP)         ---->       used by VOC PASCAL 2012)                  #
#                                                                                         #                                             #
###########################################################################################



def getBoundingBoxes(ideal, sub):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    
    allBoundingBoxes = BoundingBoxes()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # xmax, ymax represents the most bottom-right coordinates of the bounding box
    for _ ,f in ideal.iterrows():
        nameOfImage = f['image_names']
        idClass = f['cell_type'] #class
        x = float(f['xmin'])
        y = float(f['ymin'])
        w = float(f['xmax'])
        h = float(f['ymax'])
        bb = BoundingBox(nameOfImage,idClass,x,y,w,h,CoordinatesType.Absolute, (640,480), BBType.GroundTruth, format=BBFormat.XYX2Y2)
        allBoundingBoxes.addBoundingBox(bb)


    # Read detections from txt file
    # Each line of the files in the detections folder represents a detected bounding box.
    # Each value of each line is  "class_id, confidence, x, y, width, height" respectively
    # Class_id represents the class of the detected bounding box
    # Confidence represents the confidence (from 0 to 1) that this detection belongs to the class_id.
    # x, y represents the most top-left coordinates of the bounding box
    # xmax, ymax represents the most bottom-right coordinates of the bounding box
    
    for _ ,f in sub.iterrows():
        nameOfImage = f['image_names']
        idClass = f['cell_type'] #class
        confidence = f['confidence'] #probability
        x = float(f['xmin'])
        y = float(f['ymin'])
        w = float(f['xmax'])
        h = float(f['ymax'])
        bb = BoundingBox(nameOfImage,idClass,x,y,w,h,CoordinatesType.Absolute, (640,480), BBType.Detected, confidence, format=BBFormat.XYX2Y2)
        allBoundingBoxes.addBoundingBox(bb)


    return allBoundingBoxes


def calculate_score(ideal, sub):
    # Read txt files containing bounding boxes (ground truth and detections)
    boundingboxes = getBoundingBoxes(ideal, sub)

    # Create an evaluator object in order to obtain the metrics
    evaluator = Evaluator()
    ##############################################################
    # VOC PASCAL Metrics
    ##############################################################

    # Get metrics with PASCAL VOC metrics
    metricsPerClass = evaluator.GetPascalVOCMetrics(boundingboxes, # Object containing all bounding boxes (ground truths and detections)
                                                    IOUThreshold=0.3) # IOU threshold
    print("Average precision values per class:\n")
    # Loop through classes to obtain their metrics
    
    mAP = []
    for mc in metricsPerClass:
        # Get metric values per each class
        c = mc['class']
        precision = mc['precision']
        recall = mc['recall']
        average_precision = mc['AP']
        ipre = mc['interpolated precision']
        irec = mc['interpolated recall']
        # Print AP per class
        print('%s: %f' % (c, average_precision))
        mAP.append(average_precision)

    score = np.mean(mAP)
    return score