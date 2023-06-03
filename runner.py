import cv2

actual_image = cv2.imread("sample_frames/3.png")
virtual_image = cv2.imread("virtual_image.png")

actual_y, actual_x, _ = actual_image.shape
virt_y, virt_x, _ = virtual_image.shape

start_y, start_x = int((actual_y - virt_y)/2), int((actual_x - virt_x)/2)
actual_image[start_y : start_y + virt_y, start_x : start_x + virt_x] = virtual_image
cv2.imshow("Combined", actual_image)
cv2.waitKey(0) 