#graph curious_distances.npy and random_distances.npy

import matplotlib.pyplot as plt
import numpy as np

curious_distances = np.load("curious_distances.npy")
random_distances = np.load("random_distances.npy")

samples_per_iteration = 5
curious_sample_numbers = range(samples_per_iteration, len(curious_distances)*samples_per_iteration+samples_per_iteration, samples_per_iteration)
random_sample_numbers = range(samples_per_iteration, len(random_distances)*samples_per_iteration+samples_per_iteration, samples_per_iteration)

plt.plot(curious_sample_numbers, curious_distances, label="Curious")
plt.plot(random_sample_numbers, random_distances, label="Random")
plt.legend(title="Policy")
plt.xlabel("Sample Count")
plt.ylabel("Average Distance from Nearest Neighbor")
plt.title("Average Distance from Nearest Neighbor")
plt.show()
plt.clf()

curious_clap_similarites = np.load("curious_sound_distances.npy")
random_clap_similarities = np.load("random_sound_distances.npy")

plt.plot(curious_sample_numbers, curious_clap_similarites, label="Curious")
plt.plot(random_sample_numbers, random_clap_similarities, label="Random")
plt.legend(title="Policy")
plt.xlabel("Sample Count")
plt.ylabel("Average Clap Similarity")
plt.title("Average Clap Similarity Over Time")
plt.show()
plt.clf()

object_label_correctness = np.load("curious_object_accuracy.npy")
random_object_label_correctness = np.load("random_object_accuracy.npy")

plt.plot(curious_sample_numbers, object_label_correctness, label="Curious")
plt.plot(random_sample_numbers, random_object_label_correctness, label="Random")
plt.legend(title="Policy")
plt.xlabel("Sample Count")
plt.ylabel("Object Label Accuraccy")
plt.title("Object Label Accuraccy Over Time")
plt.show()
plt.clf()
