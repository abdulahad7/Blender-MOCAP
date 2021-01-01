from PyQt5 import QtWidgets, uic
import sys
import time
from oscpy.client import OSCClient
import common
import cv2
import numpy as np
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
from lifting.prob_model import Prob3dPose
from oscpy.client import OSCClient

class PoseEstimation(QtWidgets.QMainWindow):
    def __init__(self):
        super(PoseEstimation, self).__init__() # Call the inherited classes __init__ method
        uic.loadUi('ui.ui', self) # Load the .ui file
        self.show() # Show the GUI
       # _capture.clicked.connect(self.clicked())
        self._capture.clicked.connect(self.capture)
        
        self.address = "127.0.0.1"
        self.port = 5005
        self.osc = OSCClient(self.address, self.port)
        
        self.average=2
        self._average.stateChanged.connect(self.averageActive)
        self._averageValue.setText(str(self.average))
        self._averageValue.setEnabled(False)
        self._averageValue.textChanged.connect(self.syncAverage)
        
        self.threshold=80
        self._threhold.stateChanged.connect(self.thresholdActive)
        self._thresholdValue.setText(str(self.threshold))
        self._thresholdValue.setEnabled(False)
        self._thresholdValue.textChanged.connect(self.syncThreshold)

        self.fps_time = 0
        self.frame = (0, 0, 0) 
        #self.camera="./images/try.jpg"
        self.camera=0
        #self.camera= './images/a.mp4'    #     rightHAndUp // rollUp   //  sitUp   //  leftHandUp   
        self.resolution='432x368'            
        self.model='mobilenet_thin'    #cmu / mobilenet_thin
        self.poseLifting = Prob3dPose('./src/lifting/models/prob_model_params.mat')
        
        self.keypoints3d=[]
        self.previouskeypoints3d=np.array([])
        
        self.frameNumber=0
        self.cap=None    
    
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
    
    def syncAverage(self,text):
        if (text):
            self.average=text
            
        
    def debug(self,array):
        for x in range(10,len(array)):
            print(array[x])
    
    def syncThreshold(self,text):
        if (text):
            self.threhold=text
        
    def averageActive(self):
        if(self._average.isChecked()):
            self._averageValue.setEnabled(True)
            
        else:
            self._averageValue.setEnabled(False)
            
            
    
    def thresholdActive(self, text):
        if(self._threhold.isChecked()):
            self._thresholdValue.setEnabled(True)
            
        else:
            self._thresholdValue.setEnabled(False)
            
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
        
    
    
    def capture(self):
        w, h = model_wh(self.resolution)
        self.e = TfPoseEstimator(get_graph_path(self.model), target_size=(w, h))
        self.cap = cv2.VideoCapture(self.camera)
        # self.make_720p()
        # self.change_res(1280, 720)
        if (self.cap.isOpened()== False):
            print("Error opening video stream or file")
        while(self.cap.isOpened()):
            ret_val, image = self.cap.read()
            #image = cv2.rotate(image, cv2.cv2.ROTATE_90_CLOCKWISE) 
            #image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)  #for denoising
            humans = self.e.inference(image)
            image_h, image_w = image.shape[:2]
            standard_w = 640 
            standard_h = 480

            
            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
            
            cv2.putText(image,
                        "FPS: %f" % (1.0 / (time.time() - self.fps_time)),
                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
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
                
                if(self._threhold.isChecked()):
                    keypoints=self.takingThreshold(keypoints)
                    
                # if(self._threhold.isChecked()):
                #     if(self.previouskeypoints3d.size!=0):
                #         x,y=self.previouskeypoints3d.shape
                #         for i in range(17):
                #             for j in range(3):
                #                 if(self.previouskeypoints3d[i][j]+self.threshold>=keypoints[i][j] and self.previouskeypoints3d[i][j]-self.threshold<=keypoints[i][j]):
                #                     keypoints[i][j]=self.previouskeypoints3d[i][j]
                                    
                
                if(self._average.isChecked()):
                    keypoints=self.takingAverage(keypoints)

                

                self.sendingValuesToBlender(keypoints/1000)
                self.previouskeypoints3d=keypoints
                print(keypoints)
            except Exception as e:
                print(e)
                
                print("error calculating 3d estimations")
            #imS = cv2.resize(im, (960, 540))
            #image = cv2.rectangle(image, (5,5), (standard_w-5,standard_h-5), self.frame, 2)
            # self.cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
            # self.cap.set(CV_CAP_PROP_FRAME_HEIGHT, 360);

            image = cv2.rectangle(image, (5,5), (image_w-5,image_h-5), self.frame, 2)
            cv2.imshow('2d estimation', image)  
            self.fps_time = time.time()
            
            print("frameNumber " , self.frameNumber)
            self.frameNumber+=1
            if cv2.waitKey(1) == 27:
                cv2.destroyAllWindows()
                
                break
    
    def sendingValuesToBlender(self,numpyarray):
        self.osc.send_message(b'/hip', numpyarray[0])
        self.osc.send_message(b'/rHip', numpyarray[1])
        self.osc.send_message(b'/rAnkle',numpyarray[2])
        self.osc.send_message(b'/rFoot', numpyarray[3])
        self.osc.send_message(b'/lHip', numpyarray[4])
        self.osc.send_message(b'/lAnkle', numpyarray[5])
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
            
        
app = QtWidgets.QApplication(sys.argv)
window = PoseEstimation()
app.exec_()