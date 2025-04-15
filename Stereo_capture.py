import cv2

# Inisialisasi kamera kiri dan kanan
cam_left = cv2.VideoCapture(0)
cam_right = cv2.VideoCapture(1)

while True:
    retL, frameL = cam_left.read()
    retR, frameR = cam_right.read()

    if not retL or not retR:
        print("Error capturing video")
        break

    # Flip the frames (1 for horizontal, 0 for vertical, -1 for both)
    frameL = cv2.flip(frameL, 1)
    frameR = cv2.flip(frameR, 1)

    # Tampilkan hasil kamera stereo
    cv2.imshow("Left Camera", frameL) # kamera kiri
    cv2.imshow("Right Camera", frameR) # kamera kananQ

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam_left.release()
cam_right.release()
cv2.destroyAllWindows()
