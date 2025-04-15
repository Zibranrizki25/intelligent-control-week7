import cv2
import numpy as np

# Load stereo calibration parameters
stereo = cv2.StereoBM_create(numDisparities=128, blockSize=15)

def compute_depth(left_image, right_image):
    """Menghitung peta kedalaman dari dua gambar stereo"""
    grayL = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
    disparity = stereo.compute(grayL, grayR)
    return disparity

# Capture video dari dua kamera
cam_left = cv2.VideoCapture(0)
cam_right = cv2.VideoCapture(1)

while True:
    retL, frameL = cam_left.read()
    retR, frameR = cam_right.read()

    if not retL or not retR:
        break

    # Hitung kedalaman rel
    depth_map = compute_depth(frameL, frameR)

    # Tampilkan hasil
    cv2.imshow("Depth Map", depth_map)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam_left.release()
cam_right.release()
cv2.destroyAllWindows()