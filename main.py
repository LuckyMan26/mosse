# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np
from torchvision import transforms
import torch
from PIL import Image


def normalize_image(image):
    res = torch.tensor(image, dtype=torch.float32)

    res = torch.sign(res - 127.5) * torch.sqrt(torch.abs(res - 127.5))
    res -= torch.mean(res)
    res /= torch.norm(res)

    return res


def random_affine_transform(image, center_point, scale_factor_range=(0.9, 1.1), rotation_range=(-10, 10)):
    random_affine_transform = transforms.RandomAffine(degrees=(-3, 3), translate=(0.2, 0.2), scale=(0.8, 1.2))

    transformed_image = random_affine_transform(image)

    transformed_tensor = transforms.ToTensor()(transformed_image)

    return transformed_tensor


def hanning_window_2d(height, width):
    hanning_window_h = torch.hann_window(height)
    hanning_window_w = torch.hann_window(width)

    hanning_window_2d = torch.outer(hanning_window_h, hanning_window_w)

    return hanning_window_2d


class MOSSETracker(object):
    def __init__(self, roi, learning_rate, transforms_number, psr_thr, sigma):

        self.image = None
        self.G_fft = None
        self.W_fft = None
        self.g_max = None
        self.G_response = None
        self.G_response_fft = None
        self.F_fft = None
        self.F = None
        self.G = None
        self.P_transformation = transforms_number
        self.learning_rate = learning_rate

        self.roi = roi
        self.center_x = roi[2] / 2
        self.center_y = roi[3] / 2
        self.sigma_amplitude = sigma
        self.A = None
        self.B = None
        self.W = None
        self.PSR_thr = psr_thr

    def apply_mask(self, mask):
        self.F = self.F * mask.unsqueeze(0)

    def to_fft(self):
        self.F_fft = torch.fft.fft2(self.F)
        self.G_fft = torch.fft.fft2(self.G)

    def desired_response(self):
        xx, yy = np.meshgrid(np.arange(self.roi[2]), np.arange(self.roi[3]))
        center_x, center_y = self.roi[2] / 2, self.roi[3] / 2
        sigma = self.sigma_amplitude
        distance_map = (np.square(xx - center_x) + np.square(yy - center_y)) / (sigma * sigma)
        response = np.exp(-distance_map)
        response = torch.tensor(response, dtype=torch.float32)
        return response

    def train(self, image, learning_rate):
        counter = 0
        A_new = 0
        B_new = 0

        while counter < self.P_transformation:
            center = [self.roi[0] + self.roi[2] / 2, self.roi[1] + self.roi[3] / 2]
            tensor_to_pil = transforms.ToPILImage()(image)

            affine_transform = random_affine_transform(tensor_to_pil, center)

            self.F = torch.tensor(affine_transform[self.roi[1]:self.roi[1] + self.roi[3], self.roi[0]:self.roi[0] + self.roi[2]], dtype=torch.float32)

            self.F_fft = torch.fft.fft2(self.F)
            A_new += self.G_fft * self.F_fft
            B_new += self.F_fft * self.F_fft
            if learning_rate >= 1.0:
                self.A = A_new
                self.B = B_new
            else:
                self.A = learning_rate * A_new + (1 - learning_rate) * self.A
                self.B = learning_rate * B_new + (1 - learning_rate) * self.B
            self.W_fft = np.divide(self.A, self.B)
            counter += 1

    def initialize(self, image, roi):
        self.roi = roi

        self.image = normalize_image(image)
        self.G = self.desired_response()

        self.G_fft = torch.fft.fft2(self.G)
        self.train(self.image, 1.0)


    def tracking(self, frame):

        ok = True
        frame = normalize_image(frame)
        transform = transforms.ToTensor()
        prev_object_neighbourhood = frame[self.roi[1]:self.roi[1] + self.roi[3],
                                    self.roi[0]:self.roi[0] + self.roi[2]]
        F_tensor = transform(prev_object_neighbourhood)
        self.F = F_tensor
        mask = hanning_window_2d(F_tensor.shape[0], F_tensor.shape[1])

        self.apply_mask(mask)
        self.F_fft = torch.fft.fft2(self.F)
        self.G_response_fft = self.W_fft * self.F_fft
        self.G_response = torch.real(self.G_response_fft)
        max_response = self.G_response.max()
        self.g_max = np.where(self.G_response == max_response)
        x_max_position = int(np.mean(self.g_max[1]))
        y_max_position = int(np.mean(self.g_max[0]))
        dx = x_max_position - self.G_response.shape[1]
        dy = y_max_position - self.G_response.shape[0]

        PSR = (max_response - torch.std(self.G_response).item()) / torch.mean(self.G_response).item()
        if PSR >= self.PSR_thr:
            ok = True
            self.roi[0] += int(dx)
            self.roi[1] += int(dy)

            if self.roi[0] < 0:
                self.roi[0] = 0
            elif self.roi[0] >= self.image.shape[1] - self.roi[2]:
                self.roi[0] = self.image.shape[1] - self.roi[2] - 1

            if self.roi[1] < 0:
                self.roi[1] = 0
            elif self.roi[1] >= self.image.shape[0] - self.roi[3]:
                self.roi[1] = self.image.shape[0] - self.roi[3] - 1

            self.train(self.image, self.learning_rate)
        else:
            ok = False
        return self.roi, ok


if __name__ == '__main__':

    video_path = 'E:\\test.mp4'
    cap = cv2.VideoCapture(video_path)


    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()


    ret,frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Window',frame)
    tracked_object_roi = cv2.selectROI('frame', frame)
    tracker = MOSSETracker(roi=tracked_object_roi, learning_rate=0.2,
                           transforms_number=8,
                           psr_thr=5.7,
                           sigma=2)
    tracker.initialize(frame, tracked_object_roi)





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
