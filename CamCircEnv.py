from Calibration import *
from const_and_packages import *
import cv2

"""
*This class responsibility is about the execution time computation and matching between people around the circle relationship
*Used as main class for each camera that going to be executed
"""
class CamCircEnv():
    def __init__(self):
        self.looking_at_lamda = 10
        self.looking_at_array = None  # offline usage of that application
        self.current_match_iterator = 0
        self.frame_names = None
        self.length = None
        self.bins_number = 361
        self.basic_low_degree = -90
        self.basic_bin_step_degree = 0.5  # degrees step jumping in the quantization method
        self.basic_unit = None
        self.bins = None
        self.friends_ranges = None
        self.friends_tags = None
        self.my_tag = None
        self.my_degrees = None


    """
    X axis quantization
    param:
    key_points - key_points array from PoseDetector object
    friends - multi-array with friends around him with their angles from him
    mytag - the tag name you would like to give uniqely to this participator
    """
    def init_x_bins(self,key_points=None,friends=None,mytag=None):
        computed_sol = key_points[0]
        key_points = key_points[1]
        if not key_points:
            print("> Some points are not provided.")
            return None
        if not mytag:
            print("> No tag is provided.")
            return None
        else:
            self.my_tag = mytag
        if not friends:
            print("> There is initialization without some friends around you.")
        else:
            f_ranges = list()
            f_tags = list()
            for item in friends:
                f_tags.append(item[0])
                f_ranges.append(item[1])
            self.friends_ranges = f_ranges
            self.friends_tags = f_tags
        left_key_point = key_points[1][0]
        right_key_point = key_points[2][0]
        self.length = right_key_point - left_key_point + 1
        self.basic_unit = self.length/self.bins_number
        self.bins = list()
        basic_low_degree = self.basic_low_degree
        for i in range(0,self.bins_number):
            self.bins.append(basic_low_degree)
            basic_low_degree = basic_low_degree + self.basic_bin_step_degree
        self.update_my_degrees(computed_sol)
        for idx,angle_from_left in enumerate(self.friends_ranges):
            is_left = False
            if angle_from_left < 0.50:
                is_left = True
            self.friends_ranges[idx] = np.asarray(self.get_bin_num([angle_from_left,is_left]))


    """
    param:
    face_angle - friends face angle array that was created in the "init_x_bins" time
    return - quantized x axis array
    """
    def get_digitize(self,face_angle):
        ret_arr = list()
        for angle in face_angle:
            angle = float(str(angle)[:6]) #make it worth more precision tail
            quantized_location = math.floor((self.bins_number - 1)*angle)
            degree = self.bins[quantized_location]
            ret_arr.append([degree, quantized_location])
        return ret_arr


    """
    param:
    face_meta_data - my face points metadata
    return - bin location index
    """
    def get_bin_num(self,face_meta_data=None):
        if not face_meta_data:
            print("> No face meta data is provided.")
            return None
        face_angle = [face_meta_data[0]]
        bin_meta_data = self.get_digitize(face_angle)
        degrees_bias = bin_meta_data[0][0]
        return np.asarray(degrees_bias)


    """
    param:
    computed_sol - pair [percentage,is_left]
    """
    def update_my_degrees(self,computed_sol):
        self.my_degrees = np.asarray(self.get_bin_num(computed_sol))


    """
    return - array with which participate current frame looking at
    """
    def looking_at(self):
        if not self.friends_ranges:
            print("> There are no friends around you.")
            return None
        look_at_arr = list()
        for idx,f_range in enumerate(self.friends_ranges):
            is_close_to = (self.my_degrees <= (f_range + self.looking_at_lamda*self.basic_bin_step_degree)) and \
                            (self.my_degrees >= (f_range - self.looking_at_lamda*self.basic_bin_step_degree))
            if is_close_to:
                look_at_arr.append(self.friends_tags[idx])
        return look_at_arr


    """
    param:
    img - my current frame
    return - img key points from PoseDetector
    """
    def get_face_keypoints(self,img):
        return PoseDetector.get_face_keypoints(frame_name=img,main_dir=open_pose_main_dir)


    """
    param:
    frame_name - current frame name
    return - who current frame looking at
    """
    def matcher(self,frame_name=None):
        if frame_name is None:
            return None
        computed_solution = self.get_face_keypoints(frame_name)
        self.update_my_degrees(computed_solution[0])
        return self.looking_at()


    """
    param:
    each frame image that would like to compute for looking identification
    """
    def calibration(self,my_img=None,second_friend=None,third_friend=None,fourth_friend=None):
        print("> Step One:")
        print("> =========")
        print("> Calibration is running...")
        my_tag = my_img[0]
        my_frame = my_img[1]
        Calibration.make_calibration(my_img=my_img,
                                     second_friend=second_friend,
                                     third_friend=third_friend,
                                     fourth_friend=fourth_friend)
        computed_solution = self.get_face_keypoints(my_frame)
        self.init_x_bins(computed_solution, Calibration.get_calibrate(), my_tag)
        print("> Participate",my_tag,"has",self.my_degrees,"degrees of rotation")
        print("> Participates around",my_tag,"have next degrees of rotation: ")
        collect = ""
        counter = 0
        for t,r in zip(self.friends_tags,self.friends_ranges):
            collect = collect + str(t) + ":" + str(r)
            if counter != (len(self.friends_ranges) - 1):
                collect = collect + " , "
            counter = counter + 1
        print(">",collect)
        print("> Completed!")
        print("")


    def pre_matching_computation(self,iter_pre_name):
        print("> Step Two:")
        print("> =========")
        print("> Executing frame matching computations...")
        computed = list()
        names = list()
        i = 0
        try:
            while True:
                res = self.matcher(frame_name=iter_pre_name + str(i) + ".jpg")
                if res:
                    computed.append(res[0])
                else:
                    computed.append(0)
                frame_name = iter_pre_name + str(i) + ".jpg"
                names.append(frame_name)
                i = i + 1
        except:
            print("> There are",i,"computed frames that found.")
        print("> Frame detections: ")
        print(">",computed)
        print("> Frame names: ")
        print(">",names)
        print("> Completed!")
        print("")
        self.frame_names = names
        self.looking_at_array = computed


    def __is_upper_bound_iterator(self):
        return self.current_match_iterator > (len(self.looking_at_array) - 1)


    def __is_lower_bound_iterator(self):
        return self.current_match_iterator < 0


    def __is_valid_iterator(self):
        if self.__is_upper_bound_iterator() or self.__is_lower_bound_iterator():
            return False
        return True


    def get_current_frame_name(self):
        if not self.__is_valid_iterator():
            print("> No more matching is avaliable.")
            return -1
        return self.frame_names[self.current_match_iterator]

    def get_current_match(self):
        if not self.__is_valid_iterator():
            print("> No more matching is avaliable.")
            return -1
        return self.looking_at_array[self.current_match_iterator]

    def get_next_match_iterator(self):
        if self.__is_upper_bound_iterator():
            print("> No more matching is avaliable.")
            return -1
        self.current_match_iterator = self.current_match_iterator + 1

    def reset_match_iterator(self):
        self.current_match_iterator = 0





def Show_Output():      #in this funtion we show the results
    AllImSize = (1350, 700) #set up the image size
    camm1 = CamCircEnv()
    camm2 = CamCircEnv()
    camm3 = CamCircEnv()
    camm4 = CamCircEnv()

    # these images were computed already using PoseDetector than here fetch data and compute the angles as required!
    # calibration flow:

    #here we initialise the system first
    camm1.calibration(my_img=[1, "1_LookingStraight.jpg"],
                      second_friend=[2, "1_LookingAtPerson2.jpg"],
                      third_friend=[3, "1_LookingAtPerson3.jpg"],
                      fourth_friend=[4, "1_LookingAtPerson4.jpg"])
    camm2.calibration(my_img=[2, "2_LookingStraight.jpg"],
                      second_friend=[1, "2_LookingAtPerson1.jpg"],
                      third_friend=[3, "2_LookingAtPerson3.jpg"],
                      fourth_friend=[4, "2_LookingAtPerson4.jpg"])

    camm3.calibration(my_img=[3, "3_LookingStraight.jpg"],
                      second_friend=[1, "3_LookingAtPerson1.jpg"],
                      third_friend=[2, "3_LookingAtPerson2.jpg"],
                      fourth_friend=[4, "3_LookingAtPerson4.jpg"])
    camm4.calibration(my_img=[4, "4_LookingStraight.jpg"],
                      second_friend=[1, "4_LookingAtPerson1.jpg"],
                      third_friend=[2, "4_LookingAtPerson2.jpg"],
                      fourth_friend=[3, "4_LookingAtPerson3.jpg"])

    #give each person their own images path
    camm1.pre_matching_computation("1_")
    camm2.pre_matching_computation("2_")
    camm3.pre_matching_computation("3_")
    camm4.pre_matching_computation("4_")
    print("> Step Three:")

    try:
        while True:
            #here we get the face coordinates for each player
            F_Point1 = camm1.get_face_keypoints(camm1.get_current_frame_name())
            F_Point2 = camm2.get_face_keypoints(camm2.get_current_frame_name())
            F_Point3 = camm3.get_face_keypoints(camm3.get_current_frame_name())
            F_Point4 = camm4.get_face_keypoints(camm4.get_current_frame_name())

            #now we read the images for each player from the suitable file
            img1 = cv2.imread(
                'C:\\Users\\user\\Desktop\\OpenPose_demo_1.0.1\\images_buffer1\\' + str(camm1.get_current_frame_name()))
            img2 = cv2.imread(
                'C:\\Users\\user\\Desktop\\OpenPose_demo_1.0.1\\images_buffer2\\' + str(camm2.get_current_frame_name()))
            img3 = cv2.imread(
                'C:\\Users\\user\\Desktop\\OpenPose_demo_1.0.1\\images_buffer3\\' + str(camm3.get_current_frame_name()))
            img4 = cv2.imread(
                'C:\\Users\\user\\Desktop\\OpenPose_demo_1.0.1\\images_buffer4\\' + str(camm4.get_current_frame_name()))

            #here we define the rectangle coordinates
            #in order to mark the player with rectangle around his face
            UpperCoor1 = (int(F_Point1[1][1][0]), int(F_Point1[1][1][1]) - 100)
            ButtomCoor1 = (int(F_Point1[1][2][0]), int(F_Point1[1][2][1]) + 100)
            UpperCoor2 = (int(F_Point2[1][1][0]), int(F_Point2[1][1][1]) - 100)
            ButtomCoor2 = (int(F_Point2[1][2][0]), int(F_Point2[1][2][1]) + 100)
            UpperCoor3 = (int(F_Point3[1][1][0]), int(F_Point3[1][1][1]) - 100)
            ButtomCoor3 = (int(F_Point3[1][2][0]), int(F_Point3[1][2][1]) + 100)
            UpperCoor4 = (int(F_Point4[1][1][0]), int(F_Point4[1][1][1]) - 100)
            ButtomCoor4 = (int(F_Point4[1][2][0]), int(F_Point4[1][2][1]) + 100)
            line_width = 3
            green = (0, 255, 0)
            red = (0, 0, 255)
            blue = (255, 0, 0)

            #We check wether eack person is looking at
            P1_LookAt = camm1.get_current_match()  # get the number of the player that matched player 1
            P2_LookAt = camm2.get_current_match()  # get the number of the player that matched player 2
            P3_LookAt = camm3.get_current_match()  # get the number of the player that matched player 3
            P4_LookAt = camm4.get_current_match()  # get the number of the player that matched player 4

            #if we reach the end!
            if P1_LookAt == -1:     #the stop condition
                break


            #check for each two players if there is a communication between them
            if ((P1_LookAt == 2) & (P2_LookAt == 1)):
                cv2.rectangle(img1, UpperCoor1, ButtomCoor1, red, line_width)   #mark the suitable player's faces
                cv2.rectangle(img2, UpperCoor2, ButtomCoor2, red, line_width)
            if ((P1_LookAt == 3) & (P3_LookAt == 1)):
                cv2.rectangle(img1, UpperCoor1, ButtomCoor1, red, line_width)   #mark the suitable player's faces
                cv2.rectangle(img3, UpperCoor3, ButtomCoor3, red, line_width)
            if ((P1_LookAt == 4) & (P4_LookAt == 1)):
                cv2.rectangle(img1, UpperCoor1, ButtomCoor1, red, line_width)   #mark the suitable player's faces
                cv2.rectangle(img4, UpperCoor4, ButtomCoor4, red, line_width)
            if ((P2_LookAt == 3) & (P3_LookAt == 2)):
                cv2.rectangle(img3, UpperCoor3, ButtomCoor3, green, line_width) #mark the suitable player's faces
                cv2.rectangle(img2, UpperCoor2, ButtomCoor2, green, line_width)
            if ((P2_LookAt == 4) & (P4_LookAt == 2)):
                cv2.rectangle(img4, UpperCoor4, ButtomCoor4, green, line_width) #mark the suitable player's faces
                cv2.rectangle(img2, UpperCoor2, ButtomCoor2, green, line_width)
            if ((P3_LookAt == 4) & (P4_LookAt == 3)):
                cv2.rectangle(img3, UpperCoor3, ButtomCoor3, green, line_width) #mark the suitable player's faces
                cv2.rectangle(img4, UpperCoor4, ButtomCoor4, green, line_width)

            #print for each player at who he is looking
            print("P1_LookAt = " + str(P1_LookAt))
            print("P2_LookAt = " + str(P2_LookAt))
            print("P3_LookAt = " + str(P3_LookAt))
            print("P4_LookAt = " + str(P4_LookAt))

            #Here we merge all the images together in one window
            numpy_vertical_concat1 = np.concatenate((img1, img2), axis=0)
            numpy_vertical_concat2 = np.concatenate((img3, img4), axis=0)
            numpy_horizontal_concat = np.concatenate((numpy_vertical_concat1, numpy_vertical_concat2), axis=1)
            cv2.imshow('initialize', cv2.resize(numpy_horizontal_concat, AllImSize,
                                                interpolation=cv2.INTER_CUBIC))  # parameeter one: 'numpy_horizontal_concat'
            k = cv2.waitKey(1)

            #here we advance the pointer for the next frame
            camm1.get_next_match_iterator()  # ++ operator
            camm2.get_next_match_iterator()  # ++ operator
            camm3.get_next_match_iterator()  # ++ operator
            camm4.get_next_match_iterator()  # ++ operator

    except Exception as ex: #print the exception if it exist
        print(ex.with_traceback(ex.__traceback__))









def Get_input_Photos():
    #set the image's size
    AllImSize = (1350, 700)
    #set for each person his own camera
    cam1 = cv2.VideoCapture(0)
    cam2 = cv2.VideoCapture(1)
    cam3 = cv2.VideoCapture(2)
    cam4 = cv2.VideoCapture(3)

    while True:  # initializing person 1
        s, im = cam1.read() #read frame
        if not s:   #if it fails then we exit
            break
        cv2.imshow("initializing person 1", im)
        k = cv2.waitKey(1)
        if k % 256 == 27:  # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 50:  # Number two pressed
            cv2.imwrite('C:\\Users\\user\\Desktop\\OpenPose_demo_1.0.1\\images_buffer1\\1_LookingAtPerson2.jpg', im)
            cv2.waitKey(2000)
        elif k % 256 == 51:  # Number three pressed
            cv2.imwrite('C:\\Users\\user\\Desktop\\OpenPose_demo_1.0.1\\images_buffer1\\1_LookingAtPerson3.jpg', im)
            cv2.waitKey(2000)
        elif k % 256 == 52:  # Number four pressed
            cv2.imwrite('C:\\Users\\user\\Desktop\\OpenPose_demo_1.0.1\\images_buffer1\\1_LookingAtPerson4.jpg', im)
            cv2.waitKey(2000)
        elif k % 256 == 53:  # Number five pressed
            cv2.imwrite('C:\\Users\\user\\Desktop\\OpenPose_demo_1.0.1\\images_buffer1\\1_LookingStraight.jpg', im)
            cv2.waitKey(2000)

    cv2.destroyWindow('initializing person 1')

    while True:  # initializing person 2
        s, im = cam2.read()     #read frame
        if not s:   #if it fails then we exit
            break
        cv2.imshow("initializing person 2", im) # show currnet image in real-time
        k = cv2.waitKey(1)
        if k % 256 == 27:  # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 49:  # Number one pressed
            cv2.imwrite('C:\\Users\\user\\Desktop\\OpenPose_demo_1.0.1\\images_buffer2\\2_LookingAtPerson1.jpg', im)
            cv2.waitKey(2000)
        elif k % 256 == 51:  # Number three pressed
            cv2.imwrite('C:\\Users\\user\\Desktop\\OpenPose_demo_1.0.1\\images_buffer2\\2_LookingAtPerson3.jpg', im)
            cv2.waitKey(2000)
        elif k % 256 == 52:  # Number four pressed
            cv2.imwrite('C:\\Users\\user\\Desktop\\OpenPose_demo_1.0.1\\images_buffer2\\2_LookingAtPerson4.jpg', im)
            cv2.waitKey(2000)
        elif k % 256 == 53:  # Number five pressed
            cv2.imwrite('C:\\Users\\user\\Desktop\\OpenPose_demo_1.0.1\\images_buffer2\\2_LookingStraight.jpg', im)
            cv2.waitKey(2000)

    cv2.destroyWindow('initializing person 2')

    while True:  # initializing person 3
        s, im = cam3.read() #read frame
        if not s:   #if it fails then we exit
            break
        cv2.imshow("initializing person 3", im) #show current frame in real-time
        k = cv2.waitKey(1)
        if k % 256 == 27:  # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 49:  # Number one pressed
            cv2.imwrite('C:\\Users\\user\\Desktop\\OpenPose_demo_1.0.1\\images_buffer3\\3_LookingAtPerson1.jpg', im)
            cv2.waitKey(2000)
        elif k % 256 == 50:  # Number two pressed
            cv2.imwrite('C:\\Users\\user\\Desktop\\OpenPose_demo_1.0.1\\images_buffer3\\3_LookingAtPerson2.jpg', im)
            cv2.waitKey(2000)
        elif k % 256 == 52:  # Number four pressed
            cv2.imwrite('C:\\Users\\user\\Desktop\\OpenPose_demo_1.0.1\\images_buffer3\\3_LookingAtPerson4.jpg', im)
            cv2.waitKey(2000)
        elif k % 256 == 53:  # Number five pressed
            cv2.imwrite('C:\\Users\\user\\Desktop\\OpenPose_demo_1.0.1\\images_buffer3\\3_LookingStraight.jpg', im)
            cv2.waitKey(2000)

    cv2.destroyWindow('initializing person 3')

    while True:  # initializing person 4
        s, im = cam4.read()     #read frame
        if not s:   #if it fails then we exit
            break
        cv2.imshow("initializing person 4", im)
        k = cv2.waitKey(1)
        if k % 256 == 27:  # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 49:  # Number one pressed
            cv2.imwrite('C:\\Users\\user\\Desktop\\OpenPose_demo_1.0.1\\images_buffer4\\4_LookingAtPerson1.jpg', im)
            cv2.waitKey(2000)
        elif k % 256 == 50:  # Number two pressed
            cv2.imwrite('C:\\Users\\user\\Desktop\\OpenPose_demo_1.0.1\\images_buffer4\\4_LookingAtPerson2.jpg', im)
            cv2.waitKey(2000)
        elif k % 256 == 51:  # Number three pressed
            cv2.imwrite('C:\\Users\\user\\Desktop\\OpenPose_demo_1.0.1\\images_buffer4\\4_LookingAtPerson3.jpg', im)
            cv2.waitKey(2000)
        elif k % 256 == 53:  # Number five pressed
            cv2.imwrite('C:\\Users\\user\\Desktop\\OpenPose_demo_1.0.1\\images_buffer4\\4_LookingStraight.jpg', im)
            cv2.waitKey(2000)

    cv2.destroyWindow('initializing person 4')


    #Now we are going to step two
    #get the input frames and save them in the suitable folder
    PictureIndex = 0
    while True:
        #read frame for each player
        s1, im1 = cam1.read()
        s2, im2 = cam2.read()
        s3, im3 = cam3.read()
        s4, im4 = cam4.read()

        if s1:  #if person one exist
            cv2.putText(im1, "Person #1", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  #add the person's name to the current frame
            cv2.imwrite(
                'C:\\Users\\user\\Desktop\\OpenPose_demo_1.0.1\\images_buffer1\\1_' + str(PictureIndex) + '.jpg',
                im1)        #save the frame
            if s2:
                cv2.putText(im2, "Person #2", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                numpy_vertical_concat1 = np.concatenate((im1, im2), axis=0)
                cv2.imwrite(
                    'C:\\Users\\user\\Desktop\\OpenPose_demo_1.0.1\\images_buffer2\\2_' + str(PictureIndex) + '.jpg',
                    im2)
        if s3:
            cv2.putText(im3, "Person #3", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imwrite(
                'C:\\Users\\user\\Desktop\\OpenPose_demo_1.0.1\\images_buffer3\\3_' + str(PictureIndex) + '.jpg',
                im3)
            if s4:
                cv2.putText(im4, "Person #4", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                numpy_vertical_concat2 = np.concatenate((im3, im4), axis=0)
                numpy_horizontal_concat = np.concatenate((numpy_vertical_concat1, numpy_vertical_concat2), axis=1)
                cv2.imwrite(
                    'C:\\Users\\user\\Desktop\\OpenPose_demo_1.0.1\\images_buffer4\\4_' + str(PictureIndex) + '.jpg',
                    im4)

        if not s2:
            cv2.imshow('Input Channel', cv2.resize(im1, AllImSize, interpolation=cv2.INTER_CUBIC))
            cv2.waitKey(1)
        elif not s3:
            cv2.imshow('Input Channel', cv2.resize(numpy_vertical_concat1, AllImSize, interpolation=cv2.INTER_CUBIC))
            cv2.waitKey(1)
        elif not s4:
            numpy_vertical_concat2 = np.concatenate((im3, im3), axis=0)
            numpy_horizontal_concat = np.concatenate((numpy_vertical_concat1, numpy_vertical_concat2), axis=1)
            cv2.imshow('Input Channel', cv2.resize(numpy_horizontal_concat, AllImSize, interpolation=cv2.INTER_CUBIC))
        else:
            cv2.imshow('Input Channel', cv2.resize(numpy_horizontal_concat, AllImSize, interpolation=cv2.INTER_CUBIC))

        PictureIndex = PictureIndex + 1
        k = cv2.waitKey(1)
        #the stop condition
        #if ESC is pressed we exit
        if k % 256 == 27:  # ESC pressed
            print("The Input video is finished")
            break

    cv2.destroyWindow('Input Channel')


    #the last step:
    #we show the frames that we had collected
    PIndex = 0
    while True:
        if PIndex >= PictureIndex:
            break
        #read the frames in each time period
        img1 = cv2.imread('C:\\Users\\user\\Desktop\\OpenPose_demo_1.0.1\\images_buffer1\\1_' + str(PIndex) + '.jpg')
        img2 = cv2.imread('C:\\Users\\user\\Desktop\\OpenPose_demo_1.0.1\\images_buffer2\\2_' + str(PIndex) + '.jpg')
        img3 = cv2.imread('C:\\Users\\user\\Desktop\\OpenPose_demo_1.0.1\\images_buffer3\\3_' + str(PIndex) + '.jpg')
        img4 = cv2.imread('C:\\Users\\user\\Desktop\\OpenPose_demo_1.0.1\\images_buffer4\\4_' + str(PIndex) + '.jpg')

        #merge the fourth frames together in one window
        numpy_vertical_concat1 = np.concatenate((img1, img2), axis=0)
        numpy_vertical_concat2 = np.concatenate((img3, img4), axis=0)
        numpy_horizontal_concat = np.concatenate((numpy_vertical_concat1, numpy_vertical_concat2), axis=1)
        #show the input frames in one window
        cv2.imshow('initialize', cv2.resize(numpy_horizontal_concat, AllImSize, interpolation=cv2.INTER_CUBIC))
        PIndex = PIndex + 1
        cv2.waitKey(1)








if __name__ == "__main__":
    print("Please choose the suitable operation:")
    print("1) press 'i' for input images.")
    print("2) press 'o' to output the results.")
    print("3) press anything else to exit.")
    Choice = input("")

    if Choice == 'o':  #if letter 0 is pressed (output)
        Show_Output()
    elif Choice == 'i': #if letter i is pressed (input)
        Get_input_Photos()
    else:               #if other letters is pressed
        print("Exiting...")

