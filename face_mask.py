import cv2
import numpy as np
import dlib
import os
from matplotlib import pyplot as plt

path  = os.path.abspath(os.getcwd()) 
landmarks_nums = 68  # 68 or 81

# read image
img = cv2.imread(path+"/photo/chrisEvans.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(img_gray)


# detect faces
detector = dlib.get_frontal_face_detector()

# set predictor
face_landmarks = 'shape_predictor_'+str(landmarks_nums)+'_face_landmarks.dat'
predictor = dlib.shape_predictor(path+"/shapePredictor/"+face_landmarks)

faces = detector(img_gray)
imgFacelandmark = img.copy()
for face in faces:
    # detect each facelandmark
    landmarks = predictor(img_gray, face)
    landmarks_points = []
    i = 0
    for n in range(landmarks_nums):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_points.append((x, y))
        cv2.putText(imgFacelandmark, str(i), (x, y), cv2.FONT_HERSHEY_DUPLEX,
                    0.7, (255, 255, 255), 1, cv2.LINE_AA)
        i+=1
        cv2.circle(imgFacelandmark, (x, y), 5, (6, 162, 2), -1)

points = np.array(landmarks_points, np.int32)
convexhull = cv2.convexHull(points)
cv2.polylines(img, [convexhull], True, (54, 236, 52), 2)
cv2.fillConvexPoly(mask, convexhull, 255)

face_image_1 = cv2.bitwise_and(img, img, mask=mask)




# use opencv to show image

# cv2.imshow("Face", img)
# cv2.imshow("Face Mask", face_image_1)
# cv2.imshow("Mask", mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# use matplotlib to show image
titles = ["Face", "Face Mask", "Mask"]
images = [imgFacelandmark, face_image_1, mask]
plt.figure(figsize=(20, 8))
plt.axis("off")
for i in range(3):
    plt.subplot(1, 3, i+1)
    # plt.imshow(images[i], 'gray')
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
