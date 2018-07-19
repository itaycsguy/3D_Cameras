from const_and_packages import *

"""
*This class responsibility is about the calibration time before workflow is started.
*This is happening once in execution time and being as an anchor the the execution correctness and success
"""
class Calibration():
    radian2degrees = 57.2957795 # known parameter
    calibrated_array = list()

    @staticmethod
    def __dotproduct(v1, v2):
        return sum((a * b) for a, b in zip(v1, v2))

    @staticmethod
    def __length(v):
        return math.sqrt(Calibration.__dotproduct(v, v))

    @staticmethod
    def angle(v1, v2):
        return math.acos(Calibration.__dotproduct(v1, v2) / (Calibration.__length(v1) * Calibration.__length(v2)))*Calibration.radian2degrees


    """
    get all calibration frame key points, than retrieve them keypoints and compute angles between the anchor frame one ['my_img']
    ***keep the calibration array and static class array
    """
    @staticmethod
    def make_calibration(my_img=None,second_friend=None,third_friend=None,fourth_friend=None):
        if my_img is None or second_friend is None or third_friend is None or fourth_friend is None:
            return None
        Calibration.calibrated_array = list()
        second_friend_state = PoseDetector.get_face_keypoints(frame_name=second_friend[1],main_dir=open_pose_main_dir)
        Calibration.calibrated_array.append([second_friend[0],second_friend_state[0][0]])
        third_friend_state = PoseDetector.get_face_keypoints(frame_name=third_friend[1],main_dir=open_pose_main_dir)
        Calibration.calibrated_array.append([third_friend[0],third_friend_state[0][0]])
        fourth_friend_state = PoseDetector.get_face_keypoints(frame_name=fourth_friend[1],main_dir=open_pose_main_dir)
        Calibration.calibrated_array.append([fourth_friend[0],fourth_friend_state[0][0]])


    @staticmethod
    def get_calibrate():
        return Calibration.calibrated_array