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

    res = torch.sign(res - 127) * torch.sqrt(torch.abs(res - 127))
    res -= torch.mean(res)
    res /= torch.norm(res)

    return res


def hanning_window_2d(height, width):

    hanning_window_h = torch.hann_window(height)
    hanning_window_w = torch.hann_window(width)

    hanning_window_2d = torch.outer(hanning_window_h, hanning_window_w)

    return hanning_window_2d

def normalize(array):
    maximum = torch.max(array)
    minimum = torch.min(array)
    return (array - minimum) / (maximum - minimum)
def get_random_transform_mat(roi):
    amplitude = 0.025
    min_angle, max_angle = -3, 3
    angle = np.random.uniform(min_angle, max_angle)
    origin_pts = np.zeros(shape=(3, 2), dtype=np.float32)

    origin_pts[0, 0] = roi[0]
    origin_pts[0, 1] = roi[1]
    origin_pts[1, 0] = roi[0] + roi[2]
    origin_pts[1, 1] = roi[1]
    origin_pts[2, 0] = roi[0]
    origin_pts[2, 1] = roi[1] + roi[3]

    center_x = roi[0] + roi[2] / 2
    center_y = roi[1] + roi[3] / 2
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
    extended_rotation_matrix = np.concatenate((rotation_matrix, np.asarray([[0.0, 0.0, 1.0]])))

    target_pts = np.copy(origin_pts)
    affine_noise = np.asarray(np.random.uniform(-amplitude, amplitude, size=(3, 2)), dtype=np.float32)
    origin_pts[0, 0] += roi[2] * affine_noise[0, 0]
    origin_pts[0, 1] += roi[3] * affine_noise[0, 1]
    origin_pts[1, 0] += roi[2] * affine_noise[1, 0]
    origin_pts[1, 1] += roi[3] * affine_noise[1, 1]
    origin_pts[2, 0] += roi[2] * affine_noise[2, 0]
    origin_pts[2, 1] += roi[3] * affine_noise[2, 1]
    affine_matrix = cv2.getAffineTransform(origin_pts, target_pts)
    extended_affine_matrix = np.concatenate((affine_matrix, np.asarray([[0.0, 0.0, 1.0]])))
    full_transform = np.matmul(extended_rotation_matrix, extended_affine_matrix)

    return full_transform[0:2]

class MOSSETracker(object):
    def __init__(self, roi, learning_rate, transforms_number, psr_thr, sigma):

        self.image = None
        self.G_fft = None
        self.W_fft = None
        self.g_max = None



        self.G = None
        self.P_transformation = transforms_number
        self.learning_rate = learning_rate

        self.roi = roi
        self.center_x = (roi[0]+roi[2]) / 2
        self.center_y = (roi[1]+roi[3]) / 2
        self.sigma_amplitude = sigma
        self.A = None
        self.B = None
        self.W = None
        self.PSR_thr = psr_thr



    def desired_response(self):
        xx, yy = np.meshgrid(np.arange(self.roi[2]), np.arange(self.roi[3]))
        center_x, center_y = ( self.roi[2]) / 2,(self.roi[3]) / 2
        sigma = self.sigma_amplitude * np.sqrt(self.roi[2]*self.roi[3])
        distance_map = (np.square(xx - center_x) + np.square(yy - center_y)) / (sigma * sigma)
        response = np.exp(-distance_map)
        response = torch.tensor(response, dtype=torch.float32)
        response = normalize(response)
        return response


    def train(self, image, learning_rate):
        counter = 0
        A_new = 0
        B_new = 0

        f_object_image = image[self.roi[1]:self.roi[1]+self.roi[3], self.roi[0]:self.roi[0]+self.roi[2]]
        mask = hanning_window_2d(f_object_image.shape[0], f_object_image.shape[1])
        f_object_image = mask * f_object_image + (1 - mask) * torch.mean(f_object_image)
        F_object_image_freq = torch.fft.fft2(f_object_image)

        A_new = self.G_fft * (F_object_image_freq)
        B_new = F_object_image_freq * (F_object_image_freq)

        while counter < self.P_transformation:



            affine_transform_matrix = get_random_transform_mat(self.roi)
            transformed_image = cv2.warpAffine(image.numpy(), affine_transform_matrix, (image.shape[1], image.shape[0]))
            transformed_image = transformed_image[self.roi[1]:self.roi[1]+self.roi[3], self.roi[0]:self.roi[0]+self.roi[2]]

            transform = transforms.ToTensor()
            transformed_image.squeeze()
            F = transform(transformed_image)
            F = F.squeeze()


            mask = hanning_window_2d(F.shape[0], F.shape[1])

            F = mask * F + (1 - mask) * torch.mean(F)


            F_fft = torch.fft.fft2(F)

            A_new += self.G_fft * torch.conj(F_fft)
            B_new += F_fft * torch.conj(F_fft)

            if learning_rate >= 1.0:
                self.A = A_new
                self.B = B_new

            else:
                self.A = learning_rate * A_new + (1 - learning_rate) * self.A
                self.B = learning_rate * B_new + (1 - learning_rate) * self.B

            self.W_fft = torch.divide(self.A, self.B + 0.0001)
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

        prev_object_neighbourhood = frame[self.roi[1]:self.roi[1] + self.roi[3],
                                    self.roi[0]:self.roi[0] + self.roi[2]]

        F = prev_object_neighbourhood
        mask = hanning_window_2d(F.shape[0], F.shape[1])

        F = mask * F + (1 - mask) * torch.mean(F).item()

        F_fft = torch.fft.fft2(F)

        G_response_fft = self.W_fft * F_fft

        G_response = torch.abs(torch.fft.ifft2(G_response_fft))

        max_response = G_response.max()

        max_position = np.where(G_response == max_response)
        x_max_position = int(np.mean(max_position[1]))
        y_max_position = int(np.mean(max_position[0]))

        dx = x_max_position - G_response.shape[1] // 2
        dy = y_max_position - G_response.shape[0] // 2

        g_sidelobes = (G_response - self.G).numpy()
        g_sidelobes[g_sidelobes < 0] = 0.0
        g_sidelobes[y_max_position, x_max_position] = 0.0
        mean_sidelobes = np.mean(g_sidelobes)
        std_sidelobes = np.std(g_sidelobes)

        PSR = (max_response - std_sidelobes) / (mean_sidelobes + 0.0001)
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

            self.train(frame, self.learning_rate)
        else:
            ok = False
        return self.roi, ok


if __name__ == '__main__':

    video_path = 'E:\\plane.mp4'
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_file = 'E:\\video2.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height), True)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    ret,frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    tracked_object_roi = list(cv2.selectROI('frame', frame))

    tracker = MOSSETracker(roi=tracked_object_roi, learning_rate=0.2,
                           transforms_number=8,
                           psr_thr=5.7,
                           sigma=0.025)
    tracker.initialize(frame, tracked_object_roi)
    counter = 0
    while True:

        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if not ret:
            break
        roi, ok = tracker.tracking(frame)
        cv2.rectangle(frame, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (255, 0, 0), 3)
        cv2.imshow("MOSSE tracker", frame)
        out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        cv2.waitKey(1)


    out.release()
