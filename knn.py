import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import time
import pickle as pkl
import laion_clap
import numpy as np
import librosa

clap = laion_clap.CLAP_Module(enable_fusion=False)
clap.load_ckpt()
def compare_sounds(sound1: str, sound2: str):
    # Load CLAP model
    def load_and_preprocess(audio_path):
        y, sr = librosa.load(audio_path, sr=44100)
        return y, sr

    def extract_embedding(audio_path):
        audio_data, _ = librosa.load(audio_path, sr=44100)
        audio_data = audio_data.reshape(1, -1) # Make it (1,T) or (N,T)
        audio_embed = clap.get_audio_embedding_from_data(x = audio_data, use_tensor=False)
        return audio_embed

    embedding1 = extract_embedding(sound1)
    embedding2 = extract_embedding(sound2)


    similarity = np.dot(embedding1, embedding2.T) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity


def load_data(timestamp: str, data_directory="", subdirectory="", cameras = []):
    # focus_point:  <class 'tuple'>
    # focus_part:  <class 'int'>
    # xyxy:  <class 'numpy.ndarray'> (4,)
    # mask:  <class 'numpy.ndarray'> (720, 1280)
    # label:  <class 'str'>
    # DINO_embeddings:  <class 'numpy.ndarray'> (256,)
    # SAM_image_embeddings:  <class 'numpy.ndarray'> (256, 64, 64)
    # SAM_mask_embeddings:  <class 'numpy.ndarray'> (256,)
    # part_masks:  <class 'numpy.ndarray'> (9, 720, 1280)
    # part_mask_features:  <class 'numpy.ndarray'> (9, 256)
    # focus_part_image_features:  <class 'numpy.ndarray'> (256, 64, 64)
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
    if(data_directory[-1] != "/"):
        data_directory = data_directory + "/"
    
    dir = data_directory + timestamp + "/"
    if(len(cameras) == 0):
        cameras = [string for string in os.listdir(dir) if string.find(".") == -1]
    if(subdirectory != ""):
        if(subdirectory[-1] != "/"):
            subdirectory = subdirectory + "/"

    out = []
    
    for camera in cameras:
        dir = data_directory + timestamp + "/" + camera + "/"
        if(not os.path.exists(dir + subdirectory)):
            print("No subdirectory", dir + subdirectory)
            continue
        with open(dir + subdirectory + "focus_point.csv", "r") as f:
            x, y = f.read().split(",")
            focus_point = (int(x), int(y))
        with open(dir + subdirectory + "focus_object.txt", "r") as f:
            focus_object = int(f.read())
        if (focus_object == -1):
            print("No focus object!")
            print("Skipping", dir)
            continue
        with open(dir + subdirectory + "focus_part.txt", "r") as f:
            focus_part = int(f.read())
        with open(dir + subdirectory + "focus_patch_image_features.np", "rb") as f:
            focus_patch_image_features = np.load(f)
        with open(dir + "xyxy.csv", "r") as f:
            xyxy = f.read()
            xyxy = np.array([[int(float(xy.split(",")[0])), int(float(xy.split(",")[1])), int(float(xy.split(",")[2])), int(float(xy.split(",")[3]))] for xy in xyxy.split("\n") if xy != ""])
        with open(dir + "mask.np", "rb") as f:
            mask = np.load(f)
        with open(dir + "DINO_embeddings.np", "rb") as f:
            DINO_embeddings = np.load(f)
        with open(dir + "SAM_image_embeddings.np", "rb") as f:
            SAM_image_embeddings = np.load(f)
        with open(dir + "SAM_mask_embeddings.np", "rb") as f:
            SAM_mask_embeddings = np.load(f)
        with open(dir + "part_masks.pkl", "rb") as f:
            part_masks = pkl.load(f)
        with open(dir + "part_mask_features.pkl", "rb") as f:
            part_mask_features = pkl.load(f)
        with open(dir + "part_image_features.pkl", "rb") as f:
            part_image_features = pkl.load(f)
        with open(dir + "label.csv", "r") as f:
            label = f.read().split(",")
        data = {
            "focus_point": focus_point,
            "focus_object": focus_object,
            "focus_part": focus_part,
            "xyxy": xyxy,
            "mask": mask,
            "label": label,
            "DINO_embeddings": DINO_embeddings,
            "SAM_image_embeddings": SAM_image_embeddings,
            "SAM_mask_embeddings": SAM_mask_embeddings,
            "part_masks": part_masks,
            "part_mask_features": part_mask_features,
            "part_image_features": part_image_features,
            "focus_patch_image_features": focus_patch_image_features
        }
        out.append((data, int(camera[-1])))
    return out

def get_image_path(data_directory, label):
    if data_directory[-1] != "/":
        data_directory = data_directory + "/"
    ts1, ts2, camera, subdirectory = label.split("_")
    return data_directory + ts1 + "_" + ts2 + "/" + camera + "/"  "color.jpg"
    
    
def get_focus_point(data_directory, label):
    if data_directory[-1] != "/":
        data_directory = data_directory + "/"
    ts1, ts2, camera, subdirectory = label.split("_")
    with open(data_directory + ts1 + "_" + ts2 + "/" + camera + "/" + subdirectory + "/focus_point.csv", "r") as f:
        x, y = f.read().split(",")
        return (int(x), int(y))
    
def get_sound_path(data_directory, label):
    if data_directory[-1] != "/":
        data_directory = data_directory + "/"
    ts1, ts2, camera, subdirectory = label.split("_")
    return data_directory + ts1 + "_" + ts2 + "/" + subdirectory+ ".wav"
    

def load_data_label(timestamp: str, data_directory="", subdirectory=""):
    # focus_point:  <class 'tuple'>
    # focus_part:  <class 'int'>
    # xyxy:  <class 'numpy.ndarray'> (4,)
    # mask:  <class 'numpy.ndarray'> (720, 1280)
    # label:  <class 'str'>
    # DINO_embeddings:  <class 'numpy.ndarray'> (256,)
    # SAM_image_embeddings:  <class 'numpy.ndarray'> (256, 64, 64)
    # SAM_mask_embeddings:  <class 'numpy.ndarray'> (256,)
    # part_masks:  <class 'numpy.ndarray'> (9, 720, 1280)
    # part_mask_features:  <class 'numpy.ndarray'> (9, 256)
    # focus_part_image_features:  <class 'numpy.ndarray'> (256, 64, 64)
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
    if(data_directory[-1] != "/"):
        data_directory = data_directory + "/"
    
    dir = data_directory + timestamp + "/"
    cameras = [string for string in os.listdir(dir) if string.find(".") == -1]
    if(subdirectory != ""):
        if(subdirectory[-1] != "/"):
            subdirectory = subdirectory + "/"

    out = []
    
    for camera in cameras:
        dir = data_directory + timestamp + "/" + camera + "/"
        if(not os.path.exists(dir + subdirectory)):
            continue
        with open(dir + subdirectory + "focus_object.txt", "r") as f:
            focus_object = int(f.read())
        if (focus_object == -1):
            print("No focus object!")
            print("Skipping", dir)
            continue
        out.append(int(camera[-1]))
    return out

def load_dataset(data_directory, subdirectory):
    # focus_point:  <class 'tuple'>
    # focus_part:  <class 'int'>
    # xyxy:  <class 'numpy.ndarray'> (4,)
    # mask:  <class 'numpy.ndarray'> (720, 1280)
    # label:  <class 'str'>
    # DINO_embeddings:  <class 'numpy.ndarray'> (256,)
    # SAM_image_embeddings:  <class 'numpy.ndarray'> (256, 64, 64)
    # SAM_mask_embeddings:  <class 'numpy.ndarray'> (256,)
    # part_masks:  <class 'numpy.ndarray'> (9, 720, 1280)
    # part_mask_features:  <class 'numpy.ndarray'> (9, 256)
    # focus_part_image_features:  <class 'numpy.ndarray'> (256, 64, 64)
    if data_directory[-1] != "/":
        data_directory = data_directory + "/"
    if subdirectory[-1] != "/":
        subdirectory = subdirectory + "/"
    timestamps = os.listdir(data_directory)
    timestamps = np.sort(timestamps)
    labels = []
    
    for timestamp in timestamps:
        if(timestamp[:3] == "tmp"):
            continue
        sample_data = load_data_label(timestamp = timestamp, data_directory=data_directory, subdirectory=subdirectory)

        for camera in sample_data:
            label = timestamp + "_camera" +  str(camera) + "_" + subdirectory[:-1]
            labels.append(label)
    
    return labels

def load_object_label(data_directory, label):
    if data_directory[-1] != "/":
        data_directory = data_directory + "/"
    ts1, ts2, camera, subdirectory = label.split("_")
    with open(data_directory + ts1 + "_" + ts2 + "/" + camera + "/label.csv", "r") as f:
        labels = f.read().split(",")
    with open(data_directory + ts1 + "_" + ts2 + "/" + camera + "/" + subdirectory + "/focus_object.txt", "r") as f:
        focus_object = int(f.read())
    return labels[focus_object]


class KNN:
    def __init__(self, data_directory=None,  use_saved_model=False):
        self.data_directory = data_directory if data_directory is not None else "./GroundedSAM_data/"
        self.subdirectory = ""
        self.ready = False
        if(self.data_directory[-1] != '/'):
            self.data_directory = data_directory + '/'
        self.scaled_embeddings = None
        self.labels = None
        
        if(os.path.exists("knn/knn_model") and use_saved_model):
            self.classifier = pkl.load(open('knn/knn_model', 'rb'))
            # self.ready = True
        else:
            self.classifier = None

        if(os.path.exists("knn/knn_scaler") and use_saved_model):
            self.scaler = pkl.load(open('knn/knn_scaler', 'rb'))
        else:
            self.scaler = None
            
        if(os.path.exists("knn/knn_lengths") and use_saved_model):
            self.lengths = pkl.load(open('knn/knn_lengths', 'rb'))
        else:
            self.lengths = None
            
        # if not use_saved_model:
        #     self.generate_knn()
        
        

    def generate_knn(self, clear=False):
        self.ready = False
        timestamps = os.listdir(self.data_directory)
        embeddings = []
        labels = []
        lengths = [0, 0, 0, 0, 0, 0]
        if(len(timestamps) == 0):
            print("No samples found")
            self.ready = True
            return
        for timestamp in timestamps:
            if(timestamp[:3] == "tmp"):
                continue
            sample_data = load_data(timestamp = timestamp, data_directory=self.data_directory, subdirectory=self.subdirectory)
            for sample, camera in sample_data:
                focus_part = sample["focus_part"]
                focus_object = sample["focus_object"]
                dino = sample['DINO_embeddings'][focus_object].flatten()
                lengths[0] = dino.size
                sam_image = sample['SAM_image_embeddings'][focus_object].flatten()
                lengths[1] = sam_image.size
                sub_image = sample['part_image_features'][focus_object][focus_part].flatten()
                lengths[2] = sub_image.size
                sam_mask = sample['SAM_mask_embeddings'][focus_object].flatten()
                lengths[3] = sam_mask.size
                sub_mask = sample['part_mask_features'][focus_object][focus_part].flatten()
                lengths[4] = sub_mask.size
                patch_image = sample['focus_patch_image_features'].flatten()
                lengths[5] = patch_image.size
                vector = np.concatenate((dino, sam_image, sub_image, sam_mask, sub_mask, patch_image))
                label = timestamp + "_camera" +  str(camera) + "_" + self.subdirectory[:-1]
                embeddings.append(vector)
                labels.append(label)

        print("Scaling Embeddings")
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embeddings)
        max_length = max(lengths)
        index = 0
        for length in lengths:
            double_scaled = scaled_embeddings[:, index:(index+length)]
            double_scaled = double_scaled * (max_length) / length
            scaled_embeddings[:, index:(index + length)] = double_scaled
            index += length
        scaled_embeddings


        print("Fitting Classifier")
        X = scaled_embeddings
        Y = labels
        if clear or self.scaled_embeddings is None:
            self.scaled_embeddings = X
            self.labels = Y
        else:
            self.scaled_embeddings = np.vstack((self.scaled_embeddings, X))
            self.labels.extend(Y)
        classifier = KNeighborsClassifier(n_neighbors=1, weights='distance')
        classifier.fit(self.scaled_embeddings, self.labels)

        self.classifier = classifier
        self.scaler = scaler
        self.lengths = lengths
        print("Model Ready")
        self.ready = True


    def scale_embeddings(self, embeddings, scaler, lengths):
        scaled_embeddings = scaler.transform(embeddings)
        max_length = max(lengths)
        index = 0
        for length in lengths:
            double_scaled = scaled_embeddings[:, index:(index+length)]
            double_scaled = double_scaled * (max_length) / length
            scaled_embeddings[:, index:(index + length)] = double_scaled
            index += length
        return scaled_embeddings

    def preprocess_data(self, sample_data, scaler, lengths):
        focus_part = sample_data["focus_part"]
        focus_object = sample_data["focus_object"]
        dino = sample_data['DINO_embeddings'][focus_object].flatten()
        sam_image = sample_data['SAM_image_embeddings'][focus_object].flatten()
        sub_image = sample_data['part_image_features'][focus_object][focus_part].flatten()
        sam_mask = sample_data['SAM_mask_embeddings'][focus_object].flatten()
        sub_mask = sample_data['part_mask_features'][focus_object][focus_part].flatten()
        patch_image = sample_data['focus_patch_image_features'].flatten()
        
        embeddings = [np.concatenate((dino, sam_image, sub_image, sam_mask, sub_mask, patch_image))]
        scaled_embedding = self.scale_embeddings(embeddings, scaler, lengths)
        return scaled_embedding

    def predict(self, samples):
        sample_data = []
        if type(samples) == str:
            samples = [samples]
        if(type(samples[0]) == str):
            for sample in samples:
                sample_data.extend(load_data(timestamp = sample, data_directory = self.data_directory, subdirectory=self.subdirectory))
        elif(type(samples) == dict):
            sample_data = [samples]
        else:
            sample_data = samples
        
        scaled_embeddings = np.array([self.preprocess_data(datum, self.scaler, self.lengths)[0] for datum in sample_data])
        predictions = self.classifier.predict(scaled_embeddings)
        for prediction in predictions:
            print(f"Prediction: {prediction}")
        return prediction

    def get_distance(self, labels, timestamp = True):
        samples = []
        try:
            for label in labels:
                print(label)
                ts1, ts2, camera, subdirectory = label.split("_")
                samples.extend(load_data(timestamp = ts1 + "_" + ts2, data_directory = self.data_directory, subdirectory=subdirectory, cameras=[camera]))
        
            samples = [sample[0] for sample in samples]
            scaled_embeddings = np.array([self.preprocess_data(datum, self.scaler, self.lengths)[0] for datum in samples])
            dist, idxs = self.classifier.kneighbors(scaled_embeddings, return_distance=True, n_neighbors=5)
            dist = [np.mean(d) for d in dist]
            for i in range(len(dist)):
                print(f"avg distance: {dist[i]}, indexes: {idxs[i]}, labels: {[self.labels[idxs[i][j]] for j in range(len(idxs[i]))]}")
            return dist, [[self.labels[idxs[i][j]] for j in range(len(idxs[i]))] for i in range(len(idxs))]
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(e)
    
    def append_sample(self, timestamp, label=None):
        # if not # self.ready:
        #     print("Model not ready")
        #     return
        self.ready = False
        sample = None
        if(type(timestamp) == str):
            label = timestamp
            sample = load_data(timestamp = timestamp, data_directory= self.data_directory, subdirectory=self.subdirectory)
        else:
            print("Invalid input")
            self.ready = True
            return
        for sample_data, camera in sample:
            embedding = self.preprocess_data(sample_data, self.scaler, self.lengths)
            self.scaled_embeddings = np.vstack((self.scaled_embeddings, embedding))
            self.labels.append(label + "_camera" +  str(camera) + "_" + self.subdirectory[:-1])
        start = time.time()
        self.classifier.fit(self.scaled_embeddings, self.labels)
        print(f"Time to fit: {time.time() - start}")
        self.ready = True
        
    def discard_sample(self, label):
        # if not # self.ready:
        #     print("Model not ready")
        #     return
        self.ready = False
        idx = None
        for i, l in enumerate(self.labels):
            if l[0:13] == label:
                idx = i
                break
        if idx is None:
            print("Label not found")
            self.ready = True
            return
        self.scaled_embeddings = np.delete(self.scaled_embeddings, idx, axis=0)
        self.labels.pop(idx)
        self.classifier.fit(self.scaled_embeddings, self.labels)
        self.ready = True
        
    def append_from_labels(self, labels):
        if type(labels) == str:
            labels = [labels]
        self.ready = False
        embeddings = []
        # print(labels)
        for label in labels:
            # print(label)
            ts1, ts2, camera, subdirectory = label.split("_") 
            timestamp = ts1 + "_" + ts2
            sample_data = load_data(timestamp = timestamp, data_directory= self.data_directory, subdirectory=subdirectory, cameras=[camera])
            # print(camera)
            # print([sample[1] for sample in sample_data])
            for sample, camera in sample_data:
                focus_part = sample["focus_part"]
                focus_object = sample["focus_object"]
                dino = sample['DINO_embeddings'][focus_object].flatten()
                sam_image = sample['SAM_image_embeddings'][focus_object].flatten()
                sub_image = sample['part_image_features'][focus_object][focus_part].flatten()
                sam_mask = sample['SAM_mask_embeddings'][focus_object].flatten()
                sub_mask = sample['part_mask_features'][focus_object][focus_part].flatten()
                patch_image = sample['focus_patch_image_features'].flatten()
                vector = np.concatenate((dino, sam_image, sub_image, sam_mask, sub_mask, patch_image))
                embeddings.append(vector)

        # print(len(embeddings))
        scaled_embeddings = self.scale_embeddings(embeddings, self.scaler, self.lengths)
        # print(len(scaled_embeddings))

        print("Fitting Classifier")
        X = scaled_embeddings
        Y = labels
        # print(len(X))
        # print(len(Y))
        if self.scaled_embeddings is None:
            self.scaled_embeddings = X
            self.labels = list(Y)
        else:
            self.scaled_embeddings = np.vstack((self.scaled_embeddings, X))
            self.labels.extend(Y)
        self.classifier.fit(self.scaled_embeddings, self.labels)
        self.ready = True

if __name__ == "__main__":
    with open("knnpickle_file", 'rb') as f:
        knn = pkl.load(f)