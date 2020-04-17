import os
import cv2
import torch
import numpy as np
from scipy import misc


def load_test_data(image_path, size=256):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    h, w, c = img.shape
    if img.shape[2] == 4:
        white = np.ones((h, w, 3), np.uint8) * 255
        img_rgb = img[:, :, :3].copy()
        mask = img[:, :, 3].copy()
        mask = (mask / 255).astype(np.uint8)
        img = (img_rgb * mask[:, :, np.newaxis]).astype(np.uint8) + white * (1 - mask[:, :, np.newaxis])

    img = cv2.resize(img, (size, size), cv2.INTER_AREA)
    img = RGB2BGR(img)

    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)
    return img


def preprocessing(x):
    x = x/127.5 - 1
    # -1 ~ 1
    return x


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def inverse_transform(images):
    return (images+1.) / 2


def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def str2bool(x):
    return x.lower() in ('true')


def cam(x, size=256):
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (size, size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img / 255.0


def imagenet_norm(x):
    mean = [0.485, 0.456, 0.406]
    std = [0.299, 0.224, 0.225]
    mean = torch.FloatTensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)
    std = torch.FloatTensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)
    return (x - mean) / std


def denorm(x):
    return x * 0.5 + 0.5


def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1, 2, 0)


def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
