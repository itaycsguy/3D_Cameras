import os,json,sys

class PoseDetector():
	PROJ_MAIN_DIR = ".\\OpenPose_demo_1.0.1\\"		
	
	@staticmethod
	def readFramePoints(individual_tag = 1,frame_name = None):
		if individual_tag < 0:
			print("> No individual number is attached.")
			return None
		if not frame_name:
			print("> No frame is attached.")
			return None
		content = list()
		entry_directory = PoseDetector.PROJ_MAIN_DIR
		directory = entry_directory + "\\output_values\\"
		for file in os.listdir(directory):
			file_full_name = file.strip()
			try:
				file_name = str(file_full_name[:file_full_name.index("_keypoints.json")]).strip()
				target_name = str(frame_name[:str(frame_name).index(".jpg")]).strip()
				if file_name == target_name:
					content = json.load(open(directory + file_full_name,"r"))
					break
			except Exception as ex:
				print("> Parameters problem, cannot continue reading!")
		if content:
			face_items = list()
			max_x_len = 0
			for i in range(0,len(content['people'])):
				pose_values = content['people'][i]['pose_keypoints'][:]
				face_values = content['people'][i]['face_keypoints'][:]
				left_face = face_values[0:3]
				right_face = face_values[16*3:16*3+3] 
				nose = pose_values[0:3]
				x_len = right_face[0] - left_face[0]
				if x_len > max_x_len:
					face_items = [nose,left_face,right_face]
					max_x_len = x_len
			if not face_items:
				return None
			return face_items
		return None


	@staticmethod
	def convertPercentageView(points):
		x_nose = points[0][0]
		y_nose = points[0][1]
		x_left = points[1][0]
		y_left = points[1][1]
		x_right = points[2][0]
		y_right = points[2][1]
		nose_ratio = abs(x_nose-x_left)/abs(x_right-x_left)
		if x_nose >= x_right:
			nose_ratio = 1.0
		elif x_nose <= x_left:
			nose_ratio = 0.0
		return [nose_ratio,nose_ratio < 0.50]

	@staticmethod
	def detectFramePoints(individual_tag = 1,frame_path = None,percentage_output = True):
		if individual_tag < 0:
			print("> No individual number is attached.")
			return None
		if not frame_path:
			print("> No frame is attached.")
			return None
		frame_path = frame_path.split("\\")
		frame_name = frame_path[len(frame_path) - 1]
		frame_temp = ""
		for c in frame_path[:(len(frame_path) - 1)]:
			frame_temp = frame_temp + "\\" + c
		frame_path = frame_temp + "\\"
		entry_directory = PoseDetector.PROJ_MAIN_DIR
		cmd = entry_directory + "\\bin\\OpenPoseDemo.exe " + \
			   " --face " + \
               " --no_display " + \
               " --image_dir " + str(entry_directory + frame_path) + \
               " --write_keypoint_json " + entry_directory + "\\output_values\\"
		#print(cmd)
		os.system(cmd)	#takes 5 seconds to return
		if frame_name != "*":
			points = PoseDetector.readFramePoints(individual_tag,frame_name)
			if not points:
				print("> No points are detected yet.")
				return None
			if not percentage_output:
				return points
			return [PoseDetector.convertPercentageView(points),points]
			
			
	@staticmethod
	def get_face_keypoints(frame_name = None,main_dir = ""):
		if not frame_name:
			print("> No frame is provided.")
			return None
		if main_dir == "":
			print("> No OpenPose main directory is provided.")
			return None
		PoseDetector.PROJ_MAIN_DIR = main_dir
		face_points = PoseDetector.readFramePoints(frame_name=frame_name)
		if not face_points:
			print("> Cannot detect face key-points.")
			return None
		else:
			#clear points probabilities
			face_points[0] = face_points[0][:2]
			face_points[1] = face_points[1][:2]
			face_points[2] = face_points[2][:2]
			computed_points = PoseDetector.convertPercentageView(face_points)
			return [computed_points,face_points]
			

if __name__ == "__main__":
	name_arg = "*"
	if len(sys.argv) >= 3:
		name_arg = sys.argv[1] + "\\" + name_arg
		PoseDetector.PROJ_MAIN_DIR = sys.argv[2]
	computed_solution = PoseDetector.detectFramePoints(frame_path=name_arg)
	if computed_solution != None:
		print(">",computed_solution[0])