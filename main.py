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





def random_affine_transform(image, center_point, scale_factor_range=(0.9, 1.1), rotation_range=(-10, 10)):

    center = center_point


    scale_factor = np.random.uniform(scale_factor_range[0], scale_factor_range[1])
    rotation_angle = np.random.uniform(rotation_range[0], rotation_range[1])

    affine_transform = transforms.Compose([
        transforms.CenterCrop(image.size),
        transforms.Resize(image.size),
        transforms.RandomAffine(rotation_angle, center=center, scale=(scale_factor, scale_factor)),
    ])


    transformed_image = affine_transform(image)

    return transformed_image

class MOSSETracker(object):
    def __init__(self, F,G,roi):
        self.response = None
        transform = transforms.ToTensor()
        F_tensor = transform(F)
        self.F = F_tensor
        self.P_transformation = 8
        self.learning_rate = 0.25
        self.G = G
        self.roi = roi
        self.center_x = roi[2] / 2
        self.center_y = roi[3] / 2
        self.sigma_amplitude = 2
        self.A = None
        self.B = None
        self.W = None

    def apply_mask(self, mask):
        self.F  =  self.F * mask.unsqueeze(0)

    def to_fft(self):
        self.F_fft = torch.fft.fft2(self.F)
        self.G_fft = torch.fft.fft2(self.G)

    def desired_response(self):
        xx, yy = np.meshgrid(np.arange(self.roi[2]), np.arange(self.roi[3]))
        center_x, center_y = self.roi[2] / 2, self.roi[3] / 2
        sigma = self.sigma_amplitude
        distance_map = (np.square(xx - center_x) + np.square(yy - center_y)) / (sigma * sigma)
        self.response = np.exp(-distance_map)
        
    def train(self, image,eta_current):
        counter = 0
        A_new = 0
        B_new = 0
        while counter < self.P_transformation:
            A_new += self.G_fft * self.F_fft
            B_new += self.F_fft * self.F_fft
            if  self.learning_rate >= 1.0:
                self.A = A_new
                self.B = B_new
            else:
                self.A = self.learning_rate * A_new + (1 - self.learning_rate) * self.A
                self.B =  self.learning_rate * B_new + (1 - self.learning_rate ) * self.B
            self.W = np.divide(self.A, self.B)


if __name__ == '__main__':
    video_path = 'path/to/your/video.mp4'
    cap = cv2.VideoCapture(video_path)


    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    frame = cap.read()
    tracked_object_roi = cv2.selectROI('frame', frame, True, False)
    print('Hello')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
