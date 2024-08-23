import os
from knn import KNN
from geometry_msgs.msg import Point
import cv2
import numpy as np
import rospy




class UncertaintySelector:
    def __init__(self, data_directory="./knn/test_data", impatient=False):
        self.knn = KNN(data_directory=data_directory, use_saved_model=False)
        self.data_directory = data_directory
        
        self.max_distances: np.ndarray = np.array([])
        self.min_distances: np.ndarray = np.array([])
        self.patch_points: np.ndarray = np.array([])
        self.distances: np.ndarray = np.array([])
        self.norm_distances: np.ndarray = np.array([])
        self.predictions: dict = None
        self.recollect_GSAM_data = False
        self.ready = True
        
        # cv2.setMouseCallback("Color Image", self.mouse_callback)
        
        if(data_directory[-1] != '/'):   
            data_directory = data_directory + '/'
        if impatient:
            self.knn.generate_knn()

    def set_data_directory(self, data_directory, subdirectory = None):
        self.data_directory = data_directory
        if(self.data_directory[-1] != "/"):
            self.data_directory += "/"
        self.knn.data_directory = self.data_directory
        if(subdirectory is not None):
            if(subdirectory[-1] != "/"):
                subdirectory += "/"
            self.knn.subdirectory = subdirectory
    
    def color_callback_extra(self, color_image_data):
        try:
            image = np.flip(color_image_data, axis=2)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for i in range(len(self.norm_distances)):
                # Darker green = less certain, Brighter green = more certain
                if (self.patch_points[i][0] == 0):
                    cv2.circle(image, (self.patch_points[i][1], self.patch_points[i][2]), 5, (0, int(255*(1-self.norm_distances[i])), 0), -1)
            
            #TODO: Fix prediction image display
            # for i, point in enumerate(self.patch_points):
            #     if(self.patch_points[i][0] == camera.index):
            #         if(np.allclose([camera.last_click[0], camera.last_click[1]], point)):
            #             prediction_images = []
            #             for j in range(3):
            #                 prediction_image = cv2.imread(self.data_directory + self.predictions[i][j] + "/color.jpg")
            #                 path = None
            #                 if(os.path.exists(self.data_directory + self.predictions[i][j] + "/focus_point.csv")):
            #                     path = self.data_directory + self.predictions[i][j] + "/focus_point.csv"
            #                 elif(os.path.exists(self.data_directory + self.predictions[i][j] +"/" + self.knn.subdirectory + "focus_point.csv")):
            #                     path = self.data_directory + self.predictions[i][j] +"/" + self.knn.subdirectory + "focus_point.csv"
            #                 if(path is not None):
            #                     with open(path, 'r') as f:
            #                         focus_point = f.read().split(',')
            #                         cv2.circle(prediction_image, (int(focus_point[0]), int(focus_point[1])), 7, (0, 0, 255), -1)
            #                 prediction_images.append(prediction_image)
            #             prediction_images = [prediction_images[i] if prediction_images[i] is not None else np.zeros(image.shape, dtype=image.dtype) for i in range(3)]
            #             # print(self.predictions[i])
            #             # print(image.shape, )
            #             image = np.vstack((np.hstack((image, prediction_images[0])), np.hstack((prediction_images[1], prediction_images[2]))))
            #             break
            cv2.imshow("Color Image", image)
            cv2.waitKey(1)
        except Exception as e:
            print("Error in color_callback_extra")
            print(e)
            import traceback
            traceback.print_exc()
        # if key == 32:
        #     self.sample_uncertainty()
        # if key == ord('r'):
        #     self.recollect_GSAM_data = True
        #     self.sample_uncertainty()
    
    # def mouse_callback(self, event, x, y, flags, param):
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         print("Focus point set to ({}, {})".format(x, y))
    #         if(self.relay.focus_point_cv.x == -1 and self.relay.focus_point_cv.y == -1):
    #             self.relay.focus_point_cv = Point(x=x, y=y, z=0)
    #         return

    
    def build_samples(self, patch_features, GSAM_data):
        """ 
        focus_point:  <class 'tuple'>
            The focus point of the data collection. This is the point that the camera was focused on when the picture was taken.
        focus_part:  <class 'int'>
            The index of the part that the focus point is a part of.
        xyxy:  <class 'numpy.ndarray'> (4,)
            The bounding box of the object at the focus point. The format is (x1, y1, x2, y2). (top left, bottom right)
        mask:  <class 'numpy.ndarray'> (H, W)
            The mask of the object at the focus point. The mask is a binary image where 1 represents the object and 0 represents the background.
        label:  <class 'str'>
            The label of the object at the focus point.
        DINO_embeddings:  <class 'numpy.ndarray'> (256,)
            The DINO embeddings of the object at the focus point.
        SAM_image_embeddings:  <class 'numpy.ndarray'> (256, 64, 64)
            The SAM image embeddings of the object at the focus point.
        SAM_mask_embeddings:  <class 'numpy.ndarray'> (256,)
            The SAM mask embeddings of the object at the focus point.
        part_masks:  <class 'numpy.ndarray'> (n, H, W)
            The masks of the n parts in the image. The masks are binary images where 1 represents the object and 0 represents the background.
        part_mask_features:  <class 'numpy.ndarray'> (n, 256)
            The SAM mask embeddings of the n parts in the image.
        part_image_features:  <class 'numpy.ndarray'> (n, 256, 64, 64)
            The SAM image embeddings of the n parts in the image.
        focus_patch_image_features:  <class 'numpy.ndarray'> (256, 64, 64)
            The SAM image embeddings of the focus patch: A 64x64 patch of the image centered at the focus point.
        """
        samples = []
        for idx in range(patch_features["patch_points"].shape[0]):
            focus_object = -1
            focus_part = -1
            point = patch_features["patch_points"][idx]
            for i in range(len(GSAM_data["mask"])):
                if(GSAM_data["mask"][i][point[1], point[0]]):
                    focus_object = i
            
            if(focus_object == -1):
                print("Focus object not found")
                samples.append(None)
                continue
            
            for i in range(GSAM_data['part_masks'][focus_object].shape[0]).__reversed__():
                if GSAM_data['part_masks'][focus_object][i][point[1], point[0]]:
                    focus_part = i
                    break

            if focus_part == -1:
                print("Focus part not found")
                samples.append(None)
                continue
            sample = {
                "focus_point": tuple(point),
                "focus_object": focus_object,
                "focus_part": focus_part,
                "xyxy": GSAM_data['xyxy'],
                "mask": GSAM_data['mask'],
                "label": GSAM_data['label'],
                "DINO_embeddings": GSAM_data['DINO_embeddings'],
                "SAM_image_embeddings": GSAM_data['SAM_image_embeddings'],
                "SAM_mask_embeddings": GSAM_data['SAM_mask_embeddings'],
                "part_masks": GSAM_data['part_masks'],
                "part_mask_features": GSAM_data['part_mask_features'],
                "part_image_features": GSAM_data['part_image_features'],
                "focus_patch_image_features": patch_features["patch_embeddings"][idx]
            }
            samples.append(sample)
        return samples
    
    def sample_uncertainty(self, patch_features):
        print("Sampling Uncertainty")
        self.ready = False
        print("Building Samples")
        samples = self.build_samples()
        print("Filtering Points")
        valid_mask = np.array([s is not None for s in samples])
        samples = np.array(samples)
        samples = samples[valid_mask]
        index_array = np.array([[0]] * patch_features["patch_points"].shape[0])
        patch_points = np.hstack((index_array, patch_features["patch_points"]))
        patch_points = patch_points[
                            valid_mask[
                                0 * len(patch_features["patch_points"]):
                                1 * len(patch_features["patch_points"])
                            ]]
        self.patch_points = np.concatenate((self.patch_points, patch_points), axis=0) if self.patch_points.size else patch_points
        print("Predicting Distances")
        # self.knn.predict(samples)
        while(not self.knn.ready):
            print("Waiting for KNN")
            rospy.sleep(1)
        print("Getting Distances")
        dist, labels = self.knn.get_distance(samples)
        print(labels)
        print("Readying Output")
        self.distances = dist
        self.predictions = labels
        self.norm_distances = (dist-np.min(dist))/(np.max(dist)-np.min(dist))
        np.hstack((self.max_distances, (np.max(self.distances))))
        np.hstack((self.min_distances, (np.min(self.distances))))
        self.ready = True
    
def main():
    rospy.init_node('uncertainty_selector', anonymous=True)
    selector = UncertaintySelector()
    rospy.spin()