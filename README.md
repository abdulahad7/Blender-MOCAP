# 3D pose estimation for MOCAP in blender

This prototype is based on the work of tensorflow pose estimation to capture mocap. I just created the Python-to-Blender connection and rendering in Blender.

# How to use it?

* Clone [tensorflow pose estimmation](https://github.com/ildoonet/tf-pose-estimation/tree/6980660b6f50653646a33c5a493d4c51d4335a3f), follow same instructions as author says to run code.
* Install addon [addroutes](https://github.com/JPfeP/AddRoutes) for blender. 
* Put src files in tf pose estimtaion src folder.
* Open blender file.
* Now run blender_webcam python file.
* Click capture to start MOCAP in real time.

# Higher Accuracy
* If you have good GPU then use CMU model otherwise you have to run mobilenetthin model.

# TODOS
* Make gui better.
* Use [Videopose](https://github.com/facebookresearch/VideoPose3D) in future to estimate pose estimation values

# This is version 1.0 of "Realtime MOCAP with Blender". Still much things to do in future.
