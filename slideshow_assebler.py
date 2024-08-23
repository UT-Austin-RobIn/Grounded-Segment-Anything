import cv2
from knn import load_dataset, get_image_path, get_focus_point
from knn import KNN
import numpy as np


labels = load_dataset("/home/boueny/newer_data", "curious")
print("Found", len(labels), "curious labels")
print(labels)
timestamps = np.unique([label[0:13] for label in labels])
print(timestamps)
new_labels = [[]] * len(timestamps)
for idx, timestamp in enumerate(timestamps):
    new_labels[idx] = [label for label in labels if label[0:13] == timestamp]
print(new_labels)

for idx, sample in enumerate(new_labels):
    print("Found", len(sample), "samples for timestamp", timestamps[idx])
    if(len(sample) > 1):
        img1 = cv2.imread(get_image_path("/home/boueny/newer_data", sample[0]))
        hitting_point1 = get_focus_point("/home/boueny/newer_data", sample[0])
        cv2.circle(img1, hitting_point1, 10, (0, 0, 255), -1)
        img2 = cv2.imread(get_image_path("/home/boueny/newer_data", sample[1]))
        hitting_point2 = get_focus_point("/home/boueny/newer_data", sample[1])
        cv2.circle(img2, hitting_point2, 10, (0, 0, 255), -1)
        img = np.concatenate((img1, img2), axis=1)
    else:
        img = cv2.imread(get_image_path("/home/boueny/newer_data", sample[0]))
        hitting_point = get_focus_point("/home/boueny/newer_data", sample[0])
        cv2.circle(img, hitting_point, 10, (0, 0, 255), -1)
    cv2.imwrite(f"slideshow/{idx}.jpg", img)
