from PyQt5 import QtWidgets, uic
from PyQt5 import QtWidgets,QtCore, QtGui
import sys
import time
from oscpy.client import OSCClient
import common
import cv2
import numpy as np
import ui_02
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
from lifting.prob_model import Prob3dPose
from oscpy.client import OSCClient

class PoseEstimation(QtWidgets.QMainWindow):
    def __init__(self):
        super(PoseEstimation, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('./src/ui_02.ui', self) # Load the .ui file
        self.show() # Show the GUI
        
        self._address = "127.0.0.1"
        self._port = 5005
        self._threshold=80
        self._average=2
        
        self.average=2
        self.threshold=80
        
        self.fps_time = 0
        self.frame = (0, 0, 0) 
        
        self.camera=0
        self.resolution='432x368'            
        
        self.poseLifting = Prob3dPose('./src/lifting/models/prob_model_params.mat')
        
        self.keypoints3d=[]
        self.previouskeypoints3d=np.array([])
        
        self.frameNumber=0
        self.cap=None 

        self.mp4BrowseButton.clicked.connect(self.browseMP4)
        self.photoBrowseButton.clicked.connect(self.browsePhoto)
        self.startButton.clicked.connect(self.processData)
        
        self.isWebcam.stateChanged.connect(self.webcamEvent)
        self.isMP4.stateChanged.connect(self.MP4Event)
        self.isPhoto.stateChanged.connect(self.PhotoEvent)
        
        self.isThreshold.stateChanged.connect(self.isThresholdEvent)
        self.thresholdEdit.textChanged.connect(self.syncThreshold)
        self.isAverage.stateChanged.connect(self.isAverageEvent)
        self.averageEdit.textChanged.connect(self.syncAverage)

        self.setupUI()
        
    def syncAverage(self,text):
        if (text):
            self._average=text

    def make_1080p(self):
        self.cap.set(3, 1920)
        self.cap.set(4, 1080)

    def make_720p(self):
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

    def make_480p(self):
        self.cap.set(3, 640)
        self.cap.set(4, 480)

    def change_res(self,width, height):
        self.cap.set(3, width)
        self.cap.set(4, height)       
        
    def debug(self,array):
        for x in range(10,len(array)):
            print(array[x])
    
    def syncThreshold(self,text):
        if (text):
            self._threshold=text
    
    def isAverageEvent(self):
        if(self.isAverage.isChecked()):
            self.averageEdit.setEnabled(True)
        else:
            self.averageEdit.setEnabled(False)
            
    def isThresholdEvent(self):
        if(self.isThreshold.isChecked()):
            self.thresholdEdit.setEnabled(True)
        else:
            self.thresholdEdit.setEnabled(False)
        
    def browsePhoto(self):
        self.fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Single File', QtCore.QDir.rootPath())
        self.photoBrowseButton.setText(self.fileName)
        
    def browseMP4(self):
        self.fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Single File', QtCore.QDir.rootPath())
        self.mp4BrowseButton.setText(self.fileName)
        
    def webcamEvent(self):
        if self.isWebcam.isChecked():
            self.resetUi()
            self.webCameraBox(1)
            self.settingBox(1,False)
        else:
            self.settingBox(0)
            self.webCameraBox(0)
            
    def MP4Event(self):
        if self.isMP4.isChecked():
            self.resetUi()
            self.MP4Box(1)
            self.settingBox(1,False)
        else:
            self.settingBox(0)
            self.MP4Box(0)
            
    def PhotoEvent(self):
        if self.isPhoto.isChecked():
            self.resetUi()
            self.photoBox(1)
            self.settingBox(1,True)
        else:
            self.settingBox(0)
            self.photoBox(0)
    
    def resetUi(self):
        self.webCameraBox(0)
        self.MP4Box(0)
        self.photoBox(0)
        self.settingBox(0)
        
    def sendingValuesToBlender(self,numpyarray):
        self.osc = OSCClient(self._address, self._port)
        self.osc.send_message(b'/hip', numpyarray[0])
        self.osc.send_message(b'/rHip', numpyarray[1])
        self.osc.send_message(b'/rKnee',numpyarray[2])
        self.osc.send_message(b'/rFoot', numpyarray[3])
        self.osc.send_message(b'/lHip', numpyarray[4])
        self.osc.send_message(b'/lKnee', numpyarray[5])
        self.osc.send_message(b'/lFoot', numpyarray[6])
        self.osc.send_message(b'/middleSpine',numpyarray[7])
        self.osc.send_message(b'/neck', numpyarray[8])
        self.osc.send_message(b'/upperNeck', numpyarray[9])
        self.osc.send_message(b'/head', numpyarray[10])
        self.osc.send_message(b'/lShoulder', numpyarray[11])
        self.osc.send_message(b'/lElbow', numpyarray[12])
        self.osc.send_message(b'/lWrist',numpyarray[13])
        self.osc.send_message(b'/rShoulder', numpyarray[14])
        self.osc.send_message(b'/rElbow', numpyarray[15])
        self.osc.send_message(b'/rWrist', numpyarray[16])
        
    def setupUI(self):
        self.resetUi()
        self.thresholdEdit.setText(str(self._threshold))
        self.averageEdit.setText(str(self._average))
        self.adressEdit.setText(str(self._address))
        self.portEdit.setText(str(self._port))
        
        self.modelCombo.addItem("mobilenet_thin")
        self.modelCombo.addItem("cmu")
        
        self.cameraCombo.addItem("0")
        self.cameraCombo.addItem("1")
        
        self.isThreshold.setChecked(True)
        self.isAverage.setChecked(True)
        
    def webCameraBox(self,active):
            if active==0:
                self.label_2.setEnabled(False)
                self.cameraCombo.setEnabled(False)

            else:
                self.label_2.setEnabled(True)
                self.cameraCombo.setEnabled(True)
                self.isMP4.setChecked(False)
                self.isPhoto.setChecked(False)
    
    def MP4Box(self,active):
        if(active==1):

            self.label_3.setEnabled(True)
            self.mp4BrowseButton.setEnabled(True)
            self.isPhoto.setChecked(False)
            self.isWebcam.setChecked(False)
        else:

            self.label_3.setEnabled(False)
            self.mp4BrowseButton.setEnabled(False)
            
    def photoBox(self,active):
        if active==1:
            self.photoBrowseButton.setEnabled(True)
            self.label_7.setEnabled(True)
            self.isMP4.setChecked(False)
            self.isWebcam.setChecked(False)
            
        else:
            self.photoBrowseButton.setEnabled(False)
            self.label_7.setEnabled(False)
            
    def settingBox(self,active,isVideo=True):
        
        if active==1:
            self.isThreshold.setEnabled(True)
            self.thresholdEdit.setEnabled(True)
            self.isAverage.setEnabled(True)
            self.averageEdit.setEnabled(True)
            self.is2D.setEnabled(True)
            self.isDebug.setEnabled(True)
            self.isDebug.setEnabled(True)
            self.isRendering.setEnabled(True)
            self.adressEdit.setEnabled(True)
            self.portEdit.setEnabled(True)
            self.label_5.setEnabled(True)
            self.label_6.setEnabled(True)
            self.startButton.setEnabled(True)
        else:
            self.isThreshold.setEnabled(False)
            self.thresholdEdit.setEnabled(False)
            self.isAverage.setEnabled(False)
            self.averageEdit.setEnabled(False)
            self.is2D.setEnabled(False)
            self.isDebug.setEnabled(False)
            self.isDebug.setEnabled(False)
            self.isRendering.setEnabled(False)
            self.adressEdit.setEnabled(False)
            self.portEdit.setEnabled(False)
            self.label_5.setEnabled(False)
            self.label_6.setEnabled(False)
            self.startButton.setEnabled(False)
        if isVideo:
            self.isThreshold.setEnabled(False)
            self.thresholdEdit.setEnabled(False)
            self.isAverage.setEnabled(False)
            self.averageEdit.setEnabled(False)
        else:
            self.isThreshold.setEnabled(True)
            self.thresholdEdit.setEnabled(True)
            self.isAverage.setEnabled(True)
            self.averageEdit.setEnabled(True)
                    
    def processData(self):
        if self.isWebcam.isChecked():
            self.processWebcam()
        elif self.isMP4.isChecked():
            print("its a video")
            print(self.modelCombo.currentText())
        else:
            self.processPicture()

    def processPicture(self):
        w, h = model_wh(self.resolution)
        self.e = TfPoseEstimator(get_graph_path(self.modelCombo.currentText()), target_size=(w, h))

        # estimate human poses from a single image !
        
        image = common.read_imgfile(self.fileName, None, None)
        frame=image.copy()
        # image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        t = time.time()
        
        humans = self.e.inference(image)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        image_h, image_w = image.shape[:2]
        standard_w = 640
        standard_h = 480
        self.frame = (255,0,0) 
        pose_2d_mpiis = []
        visibilities = []
        try:
            for human in humans:
                pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
                pose_2d_mpiis.append([(int(x * standard_w + 0.5), int(y * standard_h + 0.5)) for x, y in pose_2d_mpii])
                visibilities.append(visibility)

            pose_2d_mpiis = np.array(pose_2d_mpiis)
            visibilities = np.array(visibilities)
            transformed_pose2d, weights = self.poseLifting.transform_joints(pose_2d_mpiis, visibilities)
            pose_3d = self.poseLifting.compute_3d(transformed_pose2d, weights)
            keypoints = pose_3d[0].transpose()
            self.sendingValuesToBlender(keypoints/600)
        except Exception as e:
            print(e)
            self.frame = (0,0,255) 
            print("error calculating 3d estimations")
        if not self.is2D.isChecked():
            if not self.isDebug.isChecked():
                self.showImage(frame)
            else:
                frame = cv2.rectangle(frame, (5,5), (image_w-5,image_h-5), self.frame, 2)
                self.showImage(frame)
        else:
            if not self.isDebug.isChecked():
                self.showImage(image)
            else:
                image = cv2.rectangle(image, (5,5), (image_w-5,image_h-5), self.frame, 2)
                self.showImage(image)
      
    def takingAverage(self,arr):
        self.keypoints3d.append(arr)
        if(len(self.keypoints3d)>self.average):
            self.keypoints3d.pop(0)
        if(len(self.keypoints3d)==self.average):
            return(np.array(self.keypoints3d).mean(axis=0))
        else:
            return arr
        
    def takingThreshold(self,keypoints):
        if(self.previouskeypoints3d.size!=0):
            x,y=self.previouskeypoints3d.shape
            for i in range(17):
                for j in range(3):
                    if(self.previouskeypoints3d[i][j]+self.threshold>=keypoints[i][j] and self.previouskeypoints3d[i][j]-self.threshold<=keypoints[i][j]):
                        keypoints[i][j]=self.previouskeypoints3d[i][j]
        return keypoints
                
    def processWebcam(self):
        w, h = model_wh(self.resolution)
        
        self.e = TfPoseEstimator(get_graph_path(self.modelCombo.currentText()), target_size=(w, h))
        self.cap = cv2.VideoCapture(int(self.cameraCombo.currentText()))

        if (self.cap.isOpened()== False):
            print("Error opening video stream or file")
        while(self.cap.isOpened()):
            ret_val, image = self.cap.read()
            frame=image.copy()
            #image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)  #for denoising
            humans = self.e.inference(image)
            image_h, image_w = image.shape[:2]
            standard_w = 640 
            standard_h = 480

            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            

            self.frame = (255,0,0) 
            try:
                pose_2d_mpiis = []
                visibilities = []
                for human in humans:
                    pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
                    pose_2d_mpiis.append([(int(x * standard_w + 0.5), int(y * standard_h + 0.5)) for x, y in pose_2d_mpii])
                    visibilities.append(visibility)

                pose_2d_mpiis = np.array(pose_2d_mpiis)
                visibilities = np.array(visibilities)
                transformed_pose2d, weights = self.poseLifting.transform_joints(pose_2d_mpiis, visibilities)
                pose_3d = self.poseLifting.compute_3d(transformed_pose2d, weights)
                keypoints = pose_3d[0].transpose()
                
                if(self.isThreshold.isChecked()):
                    keypoints=self.takingThreshold(keypoints)

                if(self.isAverage.isChecked()):
                    keypoints=self.takingAverage(keypoints)

                self.sendingValuesToBlender(keypoints)
                self.previouskeypoints3d=keypoints
                print(keypoints)
            except Exception as e:
                print(e)
                self.frame = (0,0,255) 
                print("error calculating 3d estimations")
            if not self.is2D.isChecked():
                if not self.isDebug.isChecked():
                    self.showImage(frame)
                else:
                    cv2.putText(frame,
                    "FPS: %f" % (1.0 / (time.time() - self.fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
                    frame = cv2.rectangle(frame, (5,5), (image_w-5,image_h-5), self.frame, 2)
                    self.showImage(frame)
            else:
                if not self.isDebug.isChecked():
                    self.showImage(image)
                else:
                    cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - self.fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
                    image = cv2.rectangle(image, (5,5), (image_w-5,image_h-5), self.frame, 2)
                    self.showImage(image)

            self.fps_time = time.time()
            print("frameNumber " , self.frameNumber)
            self.frameNumber+=1
            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                break              
        
        

    def showImage(self,img):
        cv2.imshow('2d estimation', img)
app = QtWidgets.QApplication(sys.argv)
window = PoseEstimation()

app.exec_()
