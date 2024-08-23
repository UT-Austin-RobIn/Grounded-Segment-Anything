from uncertainty_selector import UncertaintySelector
from knn import load_dataset, get_image_path, get_focus_point
from knn import KNN
import numpy as np
import cv2
import gc

curious_training_labels = load_dataset("/home/boueny/newer_data", "curious")
print("Found", len(curious_training_labels), "curious labels")
random_training_labels = load_dataset("/home/boueny/newer_data", "random")
print("Found", len(random_training_labels), "random labels")

unfamiliar_training_labels = sum([load_dataset("/home/boueny/newer_data", "random" + str(i)) for i in range(10)], [])

familiar = False

#Choose 25 random labels from each set to use as test data without affecting the training data

if familiar:
    curious_test_labels =np.random.choice(curious_training_labels, 0)
    random_test_labels = np.random.choice(random_training_labels , 10)

    test_labels = np.concatenate((curious_test_labels, random_test_labels))
else:
    curious_test_labels = []
    random_test_labels = []
    test_labels = unfamiliar_training_labels

#Remove the test data from the training data
curious_training_labels = np.setdiff1d(curious_training_labels, curious_test_labels)
random_training_labels = np.setdiff1d(random_training_labels, random_test_labels)

print("Curious training data:", len(curious_training_labels), "Curious test data:", len(curious_test_labels))
print("Random training data:", len(random_training_labels), "Random test data:", len(random_test_labels))


curious_knn = KNN(data_directory="/home/boueny/primer", use_saved_model=False)  
curious_knn.subdirectory = "curious"
random_knn = KNN(data_directory="/home/boueny/primer", use_saved_model=False)
random_knn.subdirectory = "random"
curious_knn.generate_knn(True)
random_knn.generate_knn(True)
curious_knn.scaled_embeddings = None
curious_knn.labels = None
random_knn.scaled_embeddings = None
random_knn.labels = None

curious_knn.data_directory = "/home/boueny/newer_data"
random_knn.data_directory = "/home/boueny/newer_data"

samples_per_iteration = 5

# print("Generating KNN from curious training data:", curious_training_labels[0:5])
curious_knn.append_from_labels(curious_training_labels[0:samples_per_iteration])
# print("Generating KNN from random training data:", random_training_labels[0:5])
random_knn.append_from_labels(random_training_labels[0:samples_per_iteration])
#Graph average distance from nearest neighbor for each point in the test data over time

curious_distances = []
random_distances = []



for i in range(samples_per_iteration, min(len(curious_training_labels), len(random_training_labels)), samples_per_iteration):
    print("Sample number:", i)
    random_knn.append_from_labels(random_training_labels[i:i+samples_per_iteration])
    
    # random_dists, _= random_knn.get_distance(test_labels, timestamp=False)
    # random_distances.append(np.mean(random_dists))
    # print ("Random:", np.mean(random_dists))
    # print("\n\n\n\n")
    
    
    image_count = 0
    np.save("random_distances.npy", random_distances)

del random_knn
gc.collect()

for i in range(samples_per_iteration, min(len(curious_training_labels), len(random_training_labels)), samples_per_iteration):
    print("Sample number:", i)
    curious_knn.append_from_labels(curious_training_labels[i:i+samples_per_iteration])
    # random_knn.append_from_labels(random_training_labels[i:i+samples_per_iteration])
    
    # curious_dists, curious_labels = curious_knn.get_distance(test_labels, timestamp=False)
    # random_dists, _= random_knn.get_distance(random_test_labels, timestamp=False)
    # print(random_dists)
    # curious_distances.append(np.mean(curious_dists))
    # random_distances.append(np.mean(random_dists))
    # print ("Curious:", np.mean(curious_dists))
    print("\n\n\n\n")

    image_count = 0
    np.save("curious_distances.npy", curious_distances)


curious_dists, curious_labels = curious_knn.get_distance(test_labels, timestamp=False)

while True:
    test_image = cv2.imread(get_image_path("/home/boueny/newer_data", test_labels[image_count]))
    test_image = cv2.resize(test_image, (0, 0), fx=0.25, fy=0.25)
    neighbor_images = [cv2.imread(get_image_path("/home/boueny/newer_data", curious_labels[image_count][i])) for i in range(3)]
    focus_point = get_focus_point("/home/boueny/newer_data", test_labels[image_count])
    neighbor_focus_points = [get_focus_point("/home/boueny/newer_data", curious_labels[image_count][i]) for i in range(3)]
    cv2.circle(test_image, focus_point, 10, (0, 0, 255), -1)
    for i in range(3):
        cv2.circle(neighbor_images[i], neighbor_focus_points[i], 10, (255, 0, 0), -1)
    #resize test image to size of neighbor images
    test_image = cv2.resize(test_image, (neighbor_images[0].shape[1], neighbor_images[0].shape[0]))
    unified_image = np.concatenate((np.concatenate((test_image, neighbor_images[0]), axis=1), np.concatenate((neighbor_images[1], neighbor_images[2]), axis=1)), axis=0)
    # shrink unified image
    unified_image = cv2.resize(unified_image, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("Curious", unified_image)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break
    elif key == ord('n'):
        image_count += 1
        if image_count >= 50:
            image_count = 0
    elif key == ord('p'):
        image_count -= 1
        if image_count < 0:
            image_count = 49
    print("Image count:", image_count)