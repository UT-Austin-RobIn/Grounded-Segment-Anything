from uncertainty_selector import UncertaintySelector
from knn import load_dataset, get_image_path, get_focus_point, get_sound_path, compare_sounds, load_object_label
from knn import KNN
import numpy as np
import cv2
import gc
from scipy.io import wavfile

curious_training_labels = load_dataset("/home/boueny/newer_data", "curious")
print("Found", len(curious_training_labels), "curious labels")
random_training_labels = load_dataset("/home/boueny/newer_data", "random")
print("Found", len(random_training_labels), "random labels")

unfamiliar_training_labels = sum([load_dataset("/home/boueny/newer_data", "random" + str(i)) for i in range(10)], [])

familiar = True

#Choose 25 random labels from each set to use as test data without affecting the training data

if familiar:
    curious_test_labels =np.random.choice(curious_training_labels, 20)
    random_test_labels = np.random.choice(random_training_labels , 20)

    test_labels = np.concatenate((curious_test_labels, random_test_labels))
else:
    curious_test_labels = []
    random_test_labels = []
    test_labels = unfamiliar_training_labels

#Remove the test data from the training data
curious_training_labels = np.setdiff1d(curious_training_labels, curious_test_labels)
random_training_labels = np.setdiff1d(random_training_labels, random_test_labels)

test_sounds = [get_sound_path("/home/boueny/newer_data", label) for label in test_labels]
test_object_labels = [load_object_label("/home/boueny/newer_data", label) for label in test_labels]


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
curious_sound_distances = []
random_distances = []
random_sound_distances = []
curious_object_accuracy = []
random_object_accuracy = []



for i in range(samples_per_iteration, min(len(curious_training_labels), len(random_training_labels)), samples_per_iteration):
    print("Sample number:", i)
    random_knn.append_from_labels(random_training_labels[i:i+samples_per_iteration])

    random_dists, random_labels = random_knn.get_distance(test_labels, timestamp=False)
    random_distances.append(np.mean(random_dists))
    
    random_sounds = [[get_sound_path("/home/boueny/newer_data", random_labels[i][j]) for j in range(len(random_labels[i]))] for i in range(len(random_labels))]
    clap_distances = [[compare_sounds(test_sounds[i], random_sounds[i][j]) for j in range(len(random_sounds[i]))] for i in range(len(random_sounds))]
    random_object_labels = [[load_object_label("/home/boueny/newer_data", random_labels[i][j]) for j in range(len(random_labels[i]))] for i in range(len(random_labels))]
    object_label_correctness = [[test_object_labels[i] == random_object_labels[i][j] for j in range(len(random_object_labels[i]))] for i in range(len(random_object_labels))]
    print("Object label correctness:", np.average(object_label_correctness))
    print("Clap distances:", np.average(clap_distances))
    random_sound_distances.append(np.average(clap_distances))
    random_object_accuracy.append(np.average(object_label_correctness))
        
    
    print ("Random:", np.mean(random_dists))
    print("\n\n\n\n")
    
    
    image_count = 0
    np.save("random_distances.npy", random_distances)
    np.save("random_sound_distances.npy", random_sound_distances)
    np.save("random_object_accuracy.npy", random_object_accuracy)

del random_knn
gc.collect()

for i in range(samples_per_iteration, min(len(curious_training_labels), len(random_training_labels)), samples_per_iteration):
    print("Sample number:", i)
    curious_knn.append_from_labels(curious_training_labels[i:i+samples_per_iteration])
    
    
    curious_dists, curious_labels = curious_knn.get_distance(test_labels, timestamp=False)
    curious_distances.append(np.mean(curious_dists))

    curious_sounds = [[get_sound_path("/home/boueny/newer_data", curious_labels[i][j]) for j in range(len(curious_labels[i]))] for i in range(len(curious_labels))]
    clap_distances = [[compare_sounds(test_sounds[i], curious_sounds[i][j]) for j in range(len(curious_sounds[i]))] for i in range(len(curious_sounds))]
    curious_object_labels = [[load_object_label("/home/boueny/newer_data", curious_labels[i][j]) for j in range(len(curious_labels[i]))] for i in range(len(curious_labels))]
    object_label_correctness = [[test_object_labels[i] == curious_object_labels[i][j] for j in range(len(curious_object_labels[i]))] for i in range(len(curious_object_labels))]
    print("Object label correctness:", np.average(object_label_correctness))
    print("Clap distances:", np.average(clap_distances))
    curious_sound_distances.append(np.average(clap_distances))
    curious_object_accuracy.append(np.average(object_label_correctness))

    print ("Curious:", np.mean(curious_dists))
    print("\n\n\n\n")

    image_count = 0
    np.save("curious_distances.npy", curious_distances)
    np.save("curious_sound_distances.npy", curious_sound_distances)
    np.save("curious_object_accuracy.npy", curious_object_accuracy)




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