import sys,numpy as np,math,os
open_pose_main_dir = None
try:
    open_pose_main_dir = os.environ['CUSTOM_OPEN_POSE']
    print(open_pose_main_dir)
    if not open_pose_main_dir:
        print("Cannot executing without 'CUSTOM_OPEN_POSE' environment variable definition.")
        sys.exit(-1)
    sys.path.append(open_pose_main_dir)
except:
    print("Cannot executing without 'CUSTOM_OPEN_POSE' environment variable definition.")
from PoseDetector import *