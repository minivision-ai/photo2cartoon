import cog
import cv2
import tempfile
import torch
import numpy as np
import os
from pathlib import Path
from utils import Preprocess
from models import ResnetGenerator


class Predictor(cog.Predictor):
    def setup(self):
        pass

    @cog.input("photo", type=Path, help="portrait photo (size < 1M)")
    def predict(self, photo):
        img = cv2.cvtColor(cv2.imread(str(photo)), cv2.COLOR_BGR2RGB)
        out_path = gen_cartoon(img)
        return out_path


def gen_cartoon(img):
    pre = Preprocess()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = ResnetGenerator(ngf=32, img_size=256, light=True).to(device)

    assert os.path.exists(
        './models/photo2cartoon_weights.pt'), "[Step1: load weights] Can not find 'photo2cartoon_weights.pt' in folder 'models!!!'"
    params = torch.load('./models/photo2cartoon_weights.pt', map_location=device)
    net.load_state_dict(params['genA2B'])

    # face alignment and segmentation
    face_rgba = pre.process(img)
    if face_rgba is None:
        return None

    face_rgba = cv2.resize(face_rgba, (256, 256), interpolation=cv2.INTER_AREA)
    face = face_rgba[:, :, :3].copy()
    mask = face_rgba[:, :, 3][:, :, np.newaxis].copy() / 255.
    face = (face * mask + (1 - mask) * 255) / 127.5 - 1

    face = np.transpose(face[np.newaxis, :, :, :], (0, 3, 1, 2)).astype(np.float32)
    face = torch.from_numpy(face).to(device)

    # inference
    with torch.no_grad():
        cartoon = net(face)[0][0]

    # post-process
    cartoon = np.transpose(cartoon.cpu().numpy(), (1, 2, 0))
    cartoon = (cartoon + 1) * 127.5
    cartoon = (cartoon * mask + 255 * (1 - mask)).astype(np.uint8)
    cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
    out_path = Path(tempfile.mkdtemp()) / "out.png"
    cv2.imwrite(str(out_path), cartoon)
    return out_path
