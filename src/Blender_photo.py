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

        
        self.address = "127.0.0.1"
        self.port = 5005
        self.osc = OSCClient(self.address, self.port)
        



        self.frame = (0, 0, 0) 
        self.camera="./images/try.jpg"

        self.resolution='432x368'            
        self.model='cmu'    #cmu / mobilenet_thin
        self.poseLifting = Prob3dPose('./src/lifting/models/prob_model_params.mat')

        self.cap=None    
    

    def capture(self):
        w, h = model_wh(self.resolution)
        self.e = TfPoseEstimator(get_graph_path(self.model), target_size=(w, h))

        # estimate human poses from a single image !
        image = common.read_imgfile(self.camera, None, None)
        # image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        t = time.time()
        humans = self.e.inference(image)
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        image_h, image_w = image.shape[:2]
        standard_w = 640
        standard_h = 480

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
        self.sendingValuesToBlender(keypoints/1000)
        print(" done with model name ",self.model)


    
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
window.capture()
app.exec_()