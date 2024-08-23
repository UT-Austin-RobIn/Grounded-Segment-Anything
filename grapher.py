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
plt.title("Average Distance from Nearest Neighbor on Unfamiliar Objects")

plt.show()