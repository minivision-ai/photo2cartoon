import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

from utils import Preprocess


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='photo folder path')
parser.add_argument('--save_path', type=str, help='save folder path')

args = parser.parse_args()
os.makedirs(args.save_path, exist_ok=True)

pre = Preprocess()

for idx, img_name in enumerate(tqdm(os.listdir(args.data_path))):
    img = cv2.cvtColor(cv2.imread(os.path.join(args.data_path, img_name)), cv2.COLOR_BGR2RGB)
    
    # face alignment and segmentation
    face_rgba = pre.process(img)
    if face_rgba is not None:
        # change background to white
        face = face_rgba[:,:,:3].copy()
        mask = face_rgba[:,:,3].copy()[:,:,np.newaxis]/255.
        face_white_bg = (face*mask + (1-mask)*255).astype(np.uint8)

        cv2.imwrite(os.path.join(args.save_path, str(idx).zfill(4)+'.png'), cv2.cvtColor(face_white_bg, cv2.COLOR_RGB2BGR))
