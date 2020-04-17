import torch
import torch.nn.functional as F
from .mobilefacenet import MobileFaceNet


class FaceFeatures(object):
    def __init__(self, weights_path, device):
        self.device = device
        self.model = MobileFaceNet(512).to(device)
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()

    def infer(self, batch_tensor):
        # crop face
        h, w = batch_tensor.shape[2:]
        top = int(h / 2.1 * (0.8 - 0.33))
        bottom = int(h - (h / 2.1 * 0.3))
        size = bottom - top
        left = int(w / 2 - size / 2)
        right = left + size
        batch_tensor = batch_tensor[:, :, top: bottom, left: right]

        batch_tensor = F.interpolate(batch_tensor, size=[112, 112], mode='bilinear', align_corners=True)

        features = self.model(batch_tensor)
        return features

    def cosine_distance(self, batch_tensor1, batch_tensor2):
        feature1 = self.infer(batch_tensor1)
        feature2 = self.infer(batch_tensor2)
        return 1 - torch.cosine_similarity(feature1, feature2)
