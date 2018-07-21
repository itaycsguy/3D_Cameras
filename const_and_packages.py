import sys,numpy as np,math,os
try:
    open_pose_main_dir = os.environ['CUSTOM_OPEN_POSE']
    sys.path.append(open_pose_main_dir)
except:
    print("Cannot executing without 'CUSTOM_OPEN_POSE' environment variable definition.")
from PoseDetector import *