# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np
from torchvision import transforms
import torch

def normalize_image(image):
    image_res =  image.astype(np.float32)
    image_res = np.sign(image_res - 127)*np.sqrt(np,abs(image_res - 127))
    # normalization
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    normalized_image = normalize(image)
    return normalized_image







class MOSSETracker(object):
    def __init__(self, F):
        transform = transforms.ToTensor()
        F_tensor = transform(F)
        self.F = F_tensor
        self.P_transformation = 8
        self.learning_rate = 0.25
    def apply_mask(self, mask):
        self.F  =  self.F * mask.unsqueeze(0)
    def to_fft(self):
        self.F_fft = torch.fft.fft2(F)

    def train(image, eta_current):
        counter = 0
        while counter < self.P_transformation:

if __name__ == '__main__':


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
