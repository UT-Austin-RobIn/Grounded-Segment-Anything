# When this function is called, take the next message from /camera/color/image_raw and save it to a file.

import random
import roslibpy
import base64
import cv2
import numpy as np
import supervision as sv

import torch
import torchvision
import torchvision.transforms as TS


from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import time
import json
from ram.models import ram
from ram import inference_ram
from PIL import Image
import pickle as pkl
import os


class GroundedSAM:
    mask_sender : roslibpy.Topic = None
    image_receiver : roslibpy.Topic = None
    focus_point_receiver : roslibpy.Topic = None
    patch_point_receiver : roslibpy.Topic = None
    datetime_receiver : roslibpy.Topic = None
    reset_receiver : roslibpy.Topic = None
    detecting_images = []
    camera = ""
    write_focus_features = True
    mask_only = False
    focus_point = (360, 240)
    detections = {}
    waiting_patch_points = []
    datetime = None
    data_directory = ""
    waiting_points = []
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    # GroundingDINO config and checkpoint
    GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"

    # Segment-Anything checkpoint
    SAM_ENCODER_VERSION = "vit_h"
    SAM_CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"
    
    # RAM checkpoint
    RAM_CHECKPOINT_PATH = "./ram_swin_large_14m.pth"

    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    
    def generate_labels(image):
        # initialize Recognize Anything Model
        IMAGE_SIZE = 384
        normalize = TS.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        transform = TS.Compose([
                        TS.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                        TS.ToTensor(), normalize
                    ])
        
        # load model
        ram_model = ram(pretrained=GroundedSAM.RAM_CHECKPOINT_PATH,
                                            image_size=IMAGE_SIZE,
                                            vit='swin_l')
        # threshold for tagging
        # we reduce the threshold to obtain more tags
        ram_model.eval()

        ram_model = ram_model.to(GroundedSAM.DEVICE)
        image_pil: Image = Image.fromarray(image)
        raw_image = image_pil.resize((IMAGE_SIZE, IMAGE_SIZE))
        raw_image  = transform(raw_image).unsqueeze(0).to(GroundedSAM.DEVICE)

        res = inference_ram(raw_image , ram_model)
        out = res[0].split(" | ")
        print("Unfiltered labels: ", out)
        # banned_labels = ["background", "table", "mat", "floor", "wall", "ceiling", "jigsaw", "puzzle", "bowling"]
        # out = [label for label in out if label != "" and np.sum([banned_label in label for banned_label in banned_labels]) == 0]
        print("Filtered labels: ", out)
        return out
    
    def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
        """Converts bounding box from xywh format to xyxy format

        Args:
            xywh (np.ndarray): The bounding box in xywh format

        Returns:
            np.ndarray: The bounding box in xyxy format
        """
        x, y, w, h = xywh
        return np.array([x, y, x + w, y + h])
    
    def detections_from_sam(sam_result):
        sorted_generated_masks = sorted(
            sam_result, key=lambda x: x["area"], reverse=True
        )
        # detections = sv.Detections()
        
        # for mask in sorted_generated_masks:
        #     detections.xyxy = np.vstack((detections.xyxy, GroundedSAM.xywh_to_xyxy(mask["bbox"])))
        #     detections.mask = np.vstack((detections.mask, mask["segmentation"]))
        #     detections.confidence = np.hstack((detections.confidence, mask["predicted_iou"]))
        #     detections.class_id = np.hstack((detections.class_id, random.randint(100, 1000000)))
        #     detections.data["features"] = np.vstack((detections.data["features"], mask["mask_embedding"]))

        return sv.Detections(
            xyxy=np.array([GroundedSAM.xywh_to_xyxy(mask["bbox"]) for mask in sorted_generated_masks]),
            mask=np.array([mask["segmentation"] for mask in sorted_generated_masks]),
            confidence=np.array([mask["predicted_iou"] for mask in sorted_generated_masks]),
            class_id=np.array([random.randint(100, 1000000) for _ in sorted_generated_masks]),
            data={"mask_features": np.array([mask["mask_embedding"] for mask in sorted_generated_masks])}
        )
    
    def auto_segment(image, old_mask=None):
        mask_generator = SamAutomaticMaskGenerator(GroundedSAM.sam)
        detections = mask_generator.generate(image)
        if old_mask is not None:
            for mask in detections:
                mask["segmentation"] = np.logical_and(mask["segmentation"], old_mask)
            print(mask.keys())
        print(f"Generated {len(detections)} masks")
        for i in range(len(detections)):
            detections[i]["area"] = np.sum(detections[i]["segmentation"])
        detections = [mask for mask in detections if mask["area"] >= 500]
        independent_masks =  np.ones((len(detections)), dtype=np.bool_)
        print(f"Filtered to {len(detections)} masks")
        # Filter out masks that overlap too much with each other
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                if not independent_masks[i] or not independent_masks[j]:
                    continue
                if np.sum(np.logical_xor(detections[i]["segmentation"], detections[j]["segmentation"])) < 150:
                    if detections[i]["area"] > detections[j]["area"]:
                        independent_masks[i] = False
                    else:
                        independent_masks[j] = False
        detections = [detections[i] for i in range(len(detections)) if independent_masks[i]]               
        
        print(f"Filtered to {len(detections)} masks")
        detections = GroundedSAM.detections_from_sam(sam_result=detections)
        # print(detections)
        return detections

    def detect(image):
        """Detect objects in an image using RAM, GroundingDINO, and SAM

        Args:
            image (np.ndarray): The image to detect objects in
            return_detections (bool, optional): _description_. Defaults to False.

        Returns:
            Dict: A dictionary containing all the relevant detection data
                "focus_point" (tuple): The focus point of the image (x, y)
                "xyxy" (np.ndarray)[4]: The bounding box of the focus object
                "mask" (np.ndarray): The mask of the focus object
                "label" (str): The label of the focus object
                "DINO_embeddings" (np.ndarray)[256]: The embeddings of the focus object generated by DINO
                "SAM_embeddings" (np.ndarray)[256]: The embeddings of the focus object generated by SAM
                "parts": A list of masks of sub-objects of the focus object
        """
        
        # Predict classes and hyper-param for GroundingDINO
        
        print("Detecting...")
        GroundedSAM.detections[GroundedSAM.camera] = None
        start_time = time.time()
        # CLASSES = GroundedSAM.generate_labels(image)
        CLASSES = ["power drill", "rubik's cube", "wood block", "peg board", "saucepan", "padlock", "scrub cleanser bottle", "can", "mug", "bowl"]
        BOX_THRESHOLD = 0.25
        TEXT_THRESHOLD = 0.25
        NMS_THRESHOLD = 0.8
        print("Classes detected: ", CLASSES)
        # detect objects
        detections = GroundedSAM.grounding_dino_model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )
        
        print(detections.data)

        # NMS post process
        print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            NMS_THRESHOLD
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        detections.data["features"] = detections.data["features"][nms_idx]
        print(f"After NMS: {len(detections.xyxy)} boxes")
                
        filter_background = np.ones((len(detections.xyxy)), dtype=np.bool_)
        
        for i in range(len(detections.xyxy)):
            w, h = detections.xyxy[i][2] - detections.xyxy[i][0], detections.xyxy[i][3] - detections.xyxy[i][1]
            if (w * h)/ (image.shape[0] * image.shape[1]) > 0.5:
                filter_background[i] = False
    
        detections.xyxy = detections.xyxy[filter_background]
        detections.confidence = detections.confidence[filter_background]
        detections.class_id = detections.class_id[filter_background]
        detections.data["features"] = detections.data["features"][filter_background]
        print("After filtering background: ", len(detections.xyxy), " boxes")
        



        # Prompting SAM with detected boxes
        def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
            sam_predictor.set_image(image)
            result_masks = []
            result_embeddings = []
            for box in xyxy:
                #TODO: Forward mask embeddings
                masks, scores, logits, mask_embeddings = sam_predictor.predict(
                    box=box,
                    multimask_output=True
                )
                index = np.argmax(scores)
                result_masks.append(masks[index])
                result_embeddings.append(mask_embeddings[index])
                
            return np.array(result_masks), np.array(result_embeddings)


        # convert detections to masks
        detections.mask, detections.data["mask_features"] = segment(
            sam_predictor=GroundedSAM.sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
        

        largest_focus = 0
        focus_mask = None
        part_detections = None
        focus_part = -1
        part_image_features = []
        

        for mask in range(detections.mask.shape[0]):
            print(f"Checking mask {mask}, object: {CLASSES[detections.class_id[mask]]}")
            if detections.mask[mask, GroundedSAM.focus_point[1], GroundedSAM.focus_point[0]]:
                print(f"Focus mask found!")
                focus_size = detections.mask[mask].sum()
                if focus_size > largest_focus:
                    largest_focus = focus_size
                    focus_mask = mask
        if(focus_mask is None):
            if(GroundedSAM.mask_only):
                GroundedSAM.detections[GroundedSAM.camera] = None
                return -1
        else:
            if GroundedSAM.mask_only:
                GroundedSAM.detections[GroundedSAM.camera] = None
                return detections.mask[focus_mask]
            
            print(f"Focus object: {CLASSES[detections.class_id[focus_mask]]}")
            print(f"Largest focus: {largest_focus}")
            # mask the image with the focus object's mask
            cropped_image = image.copy()
            cropped_image[~detections.mask[focus_mask]] = 0
            
            #crop the image to the focus object's bounding box
            x1, y1, x2, y2 =  detections.xyxy[focus_mask].astype(int)
            print(f"Focus object bounding box: {x1}, {y1}, {x2}, {y2}")
            cropped_image = cropped_image[y1:y2, x1:x2]
            GroundedSAM.sam_predictor.set_image(cropped_image)
            cropped_embeddings =  GroundedSAM.sam_predictor.get_image_embedding().cpu().numpy()[0]
            # print(f"Embeddings: {cropped_embeddings}")
            print(f"Embeddings shape: {cropped_embeddings.shape}")
            print(f"Embeddings dtype: {cropped_embeddings.dtype}")
            
            cv2.imwrite(GroundedSAM.camera + "_focus_object.jpg", cropped_image)
            part_detections = GroundedSAM.auto_segment(cropped_image, old_mask=detections.mask[focus_mask][y1:y2, x1:x2])
            #resize the masks and bounding boxes to the original image size
            part_detections.xyxy[:, 0] += x1
            part_detections.xyxy[:, 1] += y1
            part_detections.xyxy[:, 2] += x1
            part_detections.xyxy[:, 3] += y1
            resized_masks = np.zeros((part_detections.mask.shape[0], detections.mask.shape[1], detections.mask.shape[2]), dtype=np.bool_)
            for i in range(part_detections.mask.shape[0]):
                resized_masks[i, y1:y2, x1:x2] = part_detections.mask[i]
            part_detections.mask = resized_masks
            # set focus_part to the index of the focus object in the focus_detections
            if GroundedSAM.write_focus_features:
                for i in range(part_detections.mask.shape[0]):
                    if(part_detections.mask[i][GroundedSAM.focus_point[1], GroundedSAM.focus_point[0]]):
                        focus_part = i
            
            for i in range(part_detections.mask.shape[0]):
                cropped_part = image.copy()
                cropped_part[~part_detections.mask[i]] = 0
                x1, y1, x2, y2 =  part_detections.xyxy[i].astype(int)
                cropped_part = cropped_part[y1:y2, x1:x2]
                GroundedSAM.sam_predictor.set_image(cropped_part)
                cv2.imwrite(f"{GroundedSAM.camera}_focus_sub_object_{i}.jpg", cropped_part)
                part_image_features.append(GroundedSAM.sam_predictor.get_image_embedding().cpu().numpy()[0])

            if GroundedSAM.write_focus_features:
                if(GroundedSAM.focus_point[0] < 32):
                    GroundedSAM.focus_point = (32, GroundedSAM.focus_point[1])
                if(GroundedSAM.focus_point[0] > image.shape[1] - 32):
                    GroundedSAM.focus_point = (image.shape[1] - 32, GroundedSAM.focus_point[1])
                if(GroundedSAM.focus_point[1] < 32):
                    GroundedSAM.focus_point = (GroundedSAM.focus_point[0], 32)
                if(GroundedSAM.focus_point[1] > image.shape[0] - 32):
                    GroundedSAM.focus_point = (GroundedSAM.focus_point[0], image.shape[0] - 32)
                    
                patch_image = image.copy()
                print(patch_image.shape)
                print(detections.mask[focus_mask].shape)
                patch_image[~detections.mask[focus_mask]] = 0
                patch_image = patch_image[GroundedSAM.focus_point[1] - 32:GroundedSAM.focus_point[1] + 32, GroundedSAM.focus_point[0] - 32:GroundedSAM.focus_point[0] + 32]
        
                cv2.imwrite(GroundedSAM.camera + "_focus_patch.jpg", patch_image)
                GroundedSAM.sam_predictor.set_image(patch_image)
                focus_patch_image_features = GroundedSAM.sam_predictor.get_image_embedding().cpu().numpy()[0]

        # annotate image with detections
        box_annotator = sv.BoundingBoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        polygon_annotator = sv.PolygonAnnotator()

        annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections)
        if(part_detections is not None):
            annotated_image = polygon_annotator.annotate(scene=annotated_image, detections=part_detections)
        # Create a magenta circle at the focus point
        cv2.circle(annotated_image, GroundedSAM.focus_point, 5, (255, 0, 255), -1)
        cv2.imwrite(GroundedSAM.camera + "_grounded_sam_annotated_image.jpg", annotated_image)
        
        # print(detections)
        print(f"Detected {len(detections.xyxy)} objects")
        print(f"Detected {len(detections.mask)} masks")
        if(part_detections is not None):
            print(f"Detected {len(part_detections.xyxy)} focus objects")
            print(f"Detected {len(part_detections.mask)} focus masks")
        print(f"Time taken: {time.time() - start_time}")
        # print(detections.__dict__)    
        
        # def compress_masks(masks):
        #     masks = masks.astype(np.uint64)
        #     out = np.zeros((masks.shape[1], masks.shape[2]), dtype=np.uint64)
        #     assert masks.shape[0] <= 64
        #     for i in range(masks.shape[0]):
        #         out = masks[i] * i
        #     return out
        
        # detections.mask = compress_masks(detections.mask)
    
        # class Encoder(json.JSONEncoder):
        #     def default(self, obj):
        #         if isinstance(obj, np.ndarray):
        #             return obj.tolist()
        #         return json.JSONEncoder.default(self, obj)
        # #Convert detections to JSON
        # detections_json = json.dumps(detections.__dict__, cls=Encoder)
        # 
        # return detections_json
        GroundedSAM.sam_predictor.reset_image()
        if(focus_mask is not None):
            out = {}
            if GroundedSAM.write_focus_features:
                out["focus_point"] = GroundedSAM.focus_point
                out["focus_part"] = focus_part
                out["focus_patch_image_features"] = focus_patch_image_features
            
            out["xyxy"] = detections.xyxy[focus_mask]
            out["mask"] = detections.mask[focus_mask]
            out["label"] = CLASSES[detections.class_id[focus_mask]]
            out["DINO_embeddings"] = detections.data["features"][focus_mask]
            out["SAM_image_embeddings"] = cropped_embeddings
            out["SAM_mask_embeddings"] = detections.data["mask_features"][focus_mask]
            out["part_masks"] = part_detections.mask
            out["part_mask_features"] = part_detections.data["mask_features"]
            out["part_image_features"] = np.array(part_image_features)
            GroundedSAM.detections[GroundedSAM.camera] = out
            GroundedSAM.detections[GroundedSAM.camera]["image"] = image.copy()
            return out
        return -1

def get_patch_embeddings_callback(message):
    print("Received patch points")
    GroundedSAM.waiting_patch_points.append(message)
    
def process_patch_points():
    message = GroundedSAM.waiting_patch_points[0]
    GroundedSAM.waiting_patch_points = GroundedSAM.waiting_patch_points[1:]
    if(GroundedSAM.detections.get(message["camera_name"]) is None):
        GroundedSAM.waiting_patch_points.append(message)
        return
    
    if(GroundedSAM.datetime is None):
        print("No datetime set")
        return
    data_directory = "/data/" + GroundedSAM.datetime + "/" + message["camera_name"] + "/"
    points = [[int(point["x"]), int(point["y"])] for point in message["patch_points"]]
    
    if(not os.path.exists(data_directory)):
        print("Directory does not exist: ", data_directory, "\nAborting...")
        return
    
    print("Processing patch points for ", message["camera_name"])
    
    embeddings = []
    GroundedSAM.sam_predictor.reset_image()
    for point in points:
        patch_image: np.ndarray = GroundedSAM.detections[message["camera_name"]]["image"].copy()
        print(f"Getting embeddings for point {point}")
        x, y = int(point[0]), int(point[1])
        if(x < 32):
            x = 32
        if(x > patch_image.shape[1] - 32):
            x = patch_image.shape[1] - 32
        if(y < 32):
            y = 32
        if y > patch_image.shape[0] - 32:
            y = patch_image.shape[0] - 32
        patch_image[~GroundedSAM.detections[message["camera_name"]]["mask"]] = 0
        patch_image = patch_image[y - 32:y + 32, x - 32:x + 32]
        GroundedSAM.sam_predictor.set_image(patch_image)
        embeddings.append(GroundedSAM.sam_predictor.get_image_embedding().cpu().numpy()[0])
    
    embeddings = np.array(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")
    out = {
        "patch_points": np.array(points),
        "patch_embeddings": embeddings
    }
    # data = base64.b64encode(pkl.dumps(out)).decode()
    # GroundedSAM.patch_sender.publish(roslibpy.Message({"data":data}))
    # print("Published patch embeddings")
    out = pkl.dumps(out)
    try:
        with open(data_directory + "patch_features.pkl", "wb") as f:
            f.write(out)
    except Exception as e:
        print(e)
        
    print(f"Dumped patch embeddings to {data_directory}")
    

def image_callback(message):
    image = message['image']['data']
    image = base64.b64decode(image)
    mat = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    message["image"] = mat
    GroundedSAM.detecting_images.append(message)
    print("Received detection request for ", message["camera_name"])
    
    
def focus_point_callback(message):
    print(f"Received focus point: {message['camera_name']}, {message['focus_point']['x']}, {message['focus_point']['y']}")
    GroundedSAM.waiting_points.append(message)

def datetime_callback(message):
    print(f"Received datetime: {message['data']}")
    GroundedSAM.datetime = message["data"]

def dump_data(data, write_mode = None, camera = GroundedSAM.camera, directory = ""):
    if(GroundedSAM.datetime is None):
        print("No datetime set")
        return
    if(write_mode == None):
        if(GroundedSAM.write_focus_features):
            write_mode = 0
        else:
            write_mode = 1

    if(write_mode == 2):
        data_directory = "/data/" + GroundedSAM.datetime + "/" + camera + "/" + directory 
    else:
        data_directory = "/data/" + GroundedSAM.datetime + "/" + camera + "/"
    print (f"Dumping data to {data_directory}")
    
    if(type(data) is int):
        with open(data_directory + "label.txt", "w") as f:
            f.write('-1')
        return

    if(not os.path.exists(data_directory)):
        print("Directory does not exist: ", data_directory, "\nAborting...")
        return

    if(write_mode != 1):
        with open(data_directory + "focus_point.csv", "w") as f:
            f.write("{},{}".format(data['focus_point'][0], data['focus_point'][1]))
        with open(data_directory + "focus_part.txt", "w") as f:
            f.write(str(data['focus_part']))
        with open(data_directory + "focus_patch_image_features.np", "wb") as f:
            np.save(f, data['focus_patch_image_features'])
    if(write_mode != 2):
        with open(data_directory + "xyxy.csv", "w") as f:
            f.write("{},{},{},{}".format(data['xyxy'][0], data['xyxy'][1], data['xyxy'][2], data['xyxy'][3]))
        with open(data_directory + "mask.np", "wb") as f:
            np.save(f, data['mask'].astype(np.uint8))
        with open(data_directory + "DINO_embeddings.np", "wb") as f:
            np.save(f, data['DINO_embeddings'])
        with open(data_directory + "SAM_image_embeddings.np", "wb") as f:
            np.save(f, data['SAM_image_embeddings'])
        with open(data_directory + "SAM_mask_embeddings.np", "wb") as f:
            np.save(f, data['SAM_mask_embeddings'])
        with open(data_directory + "part_masks.np", "wb") as f:
            np.save(f, data['part_masks'])
        with open(data_directory + "part_mask_features.np", "wb") as f:
            np.save(f, data['part_mask_features'])
        with open(data_directory + "part_image_features.np", "wb") as f:
            np.save(f, data['part_image_features'])
        with open(data_directory + "label.txt", "w") as f:
            f.write(data['label'])
    
    print(f"Dumped data to {data_directory} with write mode {write_mode}")
    return GroundedSAM.datetime  

def process_focus_point():
    message = GroundedSAM.waiting_points[0]
    GroundedSAM.waiting_points = GroundedSAM.waiting_points[1:]

    data_directory =  message["policy_name"]
    if data_directory[-1] != "/":
        data_directory += "/"
                        
    focus_point = [int(message["focus_point"]["x"]), int(message["focus_point"]["y"])]
                        
    
    if(GroundedSAM.detections.get(message["camera_name"]) is None):
        GroundedSAM.waiting_points.append(message)
        return
                        
    out = GroundedSAM.detections[message["camera_name"]]
    out["focus_point"] = (focus_point[0], focus_point[1])
    for i in range(len(out["part_masks"])):
        if(out["part_masks"][i][focus_point[1], focus_point[0]]):
            out["focus_part"] = i
    print(f"Getting embeddings for point {focus_point}, from {message['camera_name']}")
    patch_image: np.ndarray = GroundedSAM.detections[message["camera_name"]]["image"].copy()
    x, y = int(focus_point[0]), int(focus_point[1])
    if(x < 32):
        x = 32
    if(x > patch_image.shape[1] - 32):
        x = patch_image.shape[1] - 32
    if(y < 32):
        y = 32
    if y > patch_image.shape[0] - 32:
        y = patch_image.shape[0] - 32
    patch_image[~GroundedSAM.detections[message["camera_name"]]["mask"]] = 0
    patch_image = patch_image[y - 32:y + 32, x - 32:x + 32]
    GroundedSAM.sam_predictor.set_image(patch_image)
    out["focus_patch_image_features"] = GroundedSAM.sam_predictor.get_image_embedding().cpu().numpy()[0]
    dump_data(out, write_mode=2, camera=message["camera_name"], directory=data_directory)

def compress_binary_mask(mask):
    # run-length encoding
    out = [[]] * mask.shape[0]
    for i in range(mask.shape[0]):
        unequal = mask[i][1:] != mask[i][:-1]
        idxs = np.append(np.where(unequal), mask.shape[1] - 1)
        run_lengths = np.diff(np.append(-1, idxs))
        out[i] = (run_lengths, mask[i][idxs])
    return out

def reset_callback(message):
    print("Resetting...")
    GroundedSAM.detections = {}
    GroundedSAM.detecting_images = []
    GroundedSAM.waiting_points = []
    GroundedSAM.waiting_patch_points = []
    print("Reset complete")


if __name__ == '__main__':
    client = roslibpy.Ros(host='127.0.0.1', port=9090)
    client.run()
    print('Running...')
    if client.is_connected:
        print('Connected to ROS')
    else:
        print('Failed to connect to ROS, Aborting...')
        client.terminate()
        exit(0)
    #GroundedSAM Publishers
    GroundedSAM.mask_sender = roslibpy.Topic(client, '/GroundedSAM/mask_out', 'std_msgs/String')

    
    #GroundedSAM Subscribers
    GroundedSAM.image_receiver = roslibpy.Topic(client, '/GroundedSAM/detection_request', 'gsam_msgs/DetectionRequest', queue_length=10)
    GroundedSAM.image_receiver.subscribe(image_callback)
    GroundedSAM.focus_point_receiver = roslibpy.Topic(client, '/GroundedSAM/focus_point', 'gsam_msgs/FocusPointRequest', queue_length=10)
    GroundedSAM.focus_point_receiver.subscribe(focus_point_callback)
    GroundedSAM.patch_point_receiver = roslibpy.Topic(client, '/GroundedSAM/patch_points', 'gsam_msgs/PatchPointsRequest', queue_length=10)
    GroundedSAM.patch_point_receiver.subscribe(get_patch_embeddings_callback)
    GroundedSAM.datetime_receiver = roslibpy.Topic(client, '/GroundedSAM/datetime', 'std_msgs/String')
    GroundedSAM.datetime_receiver.subscribe(datetime_callback)
    GroundedSAM.reset_receiver = roslibpy.Topic(client, '/GroundedSAM/reset', 'std_msgs/Empty')
    GroundedSAM.reset_receiver.subscribe(reset_callback)

    timeout_counter = 0
    try:
        while True:
            if(client.is_connected):
                timeout_counter = 0
                if(len(GroundedSAM.detecting_images) > 0):
                    message = GroundedSAM.detecting_images[0]
                    GroundedSAM.detecting_images = GroundedSAM.detecting_images[1:]
                    GroundedSAM.mask_only = message["mask_only"]
                    GroundedSAM.focus_point = [int(message["focus_point"]["x"]), int(message["focus_point"]["y"])]
                    GroundedSAM.write_focus_features = message["write_focus_features"]
                    GroundedSAM.camera = message["camera_name"]
                    print(f"Detecting for camera: {GroundedSAM.camera}")
                    out = GroundedSAM.detect(message["image"])
                    if (GroundedSAM.mask_only):
                        GroundedSAM.mask_only = False
                        if(type(out) == int):
                            mask = -1
                        else:
                            mask = compress_binary_mask(out)
                        mask_bytes = base64.b64encode(pkl.dumps(mask)).decode()
                        print("Mask compressed by a factor of: ", len(base64.b64encode(pkl.dumps(out)).decode())/len(mask_bytes))
                        GroundedSAM.mask_sender.publish(roslibpy.Message({"data": mask_bytes}))
                        continue
                    # encode out as a string
                    dump_data(out, camera=message["camera_name"])
                    print("Published!\n\n\n\n")
                if(len(GroundedSAM.waiting_patch_points) > 0):
                    process_patch_points()
                if(len(GroundedSAM.waiting_points) > 0):
                    process_focus_point()
                time.sleep(0.2)
            elif timeout_counter < 120:
                timeout_counter += 1
                time.sleep(5)
                print(f"Lost connection to ROS, attempting to reconnect: Attempt # {timeout_counter}")
            else:
                "Failed to reconnect to ROS, Aborting..."
                client.terminate()
                print("Execution Terminated")
    except KeyboardInterrupt:
        client.terminate()
        print("Execution Terminated")