import cv2

def show_image(image):
    cv2.imshow("Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
