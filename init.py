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

from connect_complete import GroundedSAM