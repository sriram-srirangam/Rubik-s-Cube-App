import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import transform as tf

# og_img = io
og_img = cv2.cvtColor(cv2.imread('left.jpg'), cv2.COLOR_BGR2RGB)
img = cv2.imread('left.jpg', 0)
edges = cv2.Canny(img[:3200, :2100],300,300)


# plt.subplot(121),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.show()

indices = np.where(edges != [0])
coordinates = zip(indices[0], indices[1])
# print(list(coordinates))
x_list = list(indices[0])
y_list = list(indices[1])
x_min = min(x_list)
y_min = min(y_list)
x_max = max(x_list)
y_max = max(y_list)
top_left = x_min, y_min
top_right = x_max, y_min
bottom_left = x_min, y_max
bottom_right = x_max, y_max


plt.figure()
plt.imshow(og_img)
plt.plot((y_min, y_min, y_max, y_max, y_min), (x_min, x_max, x_max, x_min, x_min), 'r')
plt.show()

y = [x_min, x_max, x_max, x_min]
x = [y_min, y_min, y_max, y_max]

print("x", x)
print("y", y)

# 4 pixels per mm
f = 4
rubiks_mm = 57

# Dimensions of the bill are 150 mm x 70 mm
x2 = [0+1, rubiks_mm*f+1, rubiks_mm*f+1, 0+1]
y2 = [0+1, 0+1, rubiks_mm*f+1, rubiks_mm*f+1]

src = np.vstack((x, y)).T
dst = np.vstack((x2, y2)).T
tform = tf.estimate_transform('projective', src, dst)

warped = tf.warp(og_img, inverse_map=tform.inverse, output_shape=(rubiks_mm * f, rubiks_mm * f))

# Transform the corners of the door and the bill using the parameters we found
# transformed = tform.params.dot(np.row_stack((x, y, [1] * 4)))
# for i in range(4):
#     transformed[:, i] = transformed[:, i] / transformed[2, i]

# transformed_h_x = list(transformed[0, :])
# transformed_h_y = list(transformed[1, :])

plt.figure()
plt.imshow(warped)
plt.plot(x2 + x2[:1], y2 + y2[:1], 'r-')
plt.plot([19*f, 19*f], [1, 57*f], 'r-')
plt.plot([38*f, 38*f], [1, 57*f], 'r-')
plt.plot([1, 57*f], [19*f, 19*f], 'r-')
plt.plot([1, 57*f], [38*f, 38*f], 'r-')
plt.show()
