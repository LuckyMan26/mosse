# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np
def normalize_image(image):
    image_res =  image.astype(np.float32)
    image_res = np.sign(image_res - 127)*np.sqrt(np,abs(image_res - 127))
    # normalization
    image_res -= np.mean(image)
    image /= np.linalg.norm(image)
    return image


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
