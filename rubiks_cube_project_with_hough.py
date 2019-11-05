import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import transform as tf
from scipy.cluster.vq import vq, kmeans, whiten
from rubik_solver import utils

# Define global strings for image reading
sides = ["top", "left", "front", "right", "back", "bottom"]
dataset = "7t"

# Set this global flag True to see intermediate plots
plot_intermediate_results = True

# Find the corners of the cube in the given image
def get_corners_canny(img_name):
    img = cv2.imread(img_name, 0)
    edges = cv2.Canny(img, 300, 300)

    indices = np.where(edges != [0])
    coordinates = zip(indices[0], indices[1])
    x_list = list(indices[0])
    y_list = list(indices[1])
    x_min = min(x_list)
    y_min = min(y_list)
    x_max = max(x_list)
    y_max = max(y_list)

    x = [x_min, x_max, x_max, x_min]
    y = [y_min, y_min, y_max, y_max]

    # Plot the detected cube
    if plot_intermediate_results:
        plt.figure()
        plt.imshow(cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB))
        plt.plot(y + y[:1], x + x[:1], 'r')
        plt.title("Detected " + img_name)
        plt.show()

    return x, y

def get_corners_hough(img_name):
    img = cv2.imread(img_name)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 120, apertureSize = 3)

    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)

    def add_line_to_img(img, m, b):
        if m != np.inf:
            xl = 0
            xr = img.shape[1] - 1
            yl = int(m * xl + b)
            yr = int(m * xr + b)
        else:
            xl = xr = b
            yl = 0
            yr = img.shape[0] - 1

        cv2.line(img, (xl,yl), (xr,yr), (0,255,0), 2)

    def are_perp(m1, m2):
        if m1 != 0:
            eps = 0.02
            return (-1 / m1) - eps <= m2 <= (-1 / m1) + eps
        else:
            return m2 == np.inf

    def solve_for_intersections(m1, b1, m2, b2):
        if m1 == 0:
            return (b2, b1)
        elif m1 == np.inf:
            return (b1, b2)
        else:
            x = (b2 - b1) / (m1 - m2)
            y = m1 * x + b1
            return (x, y)

    def find_corners(corners):
        corners_x = [corner[0] for corner in corners]
        corners_y = [corner[1] for corner in corners]

        if corners == []:
            return [], []

        min_x = min(corners_x)
        max_x = max(corners_x)
        min_y = min(corners_y)
        max_y = max(corners_y)

        eps = 5
        min_x_corners = [corner for corner in corners if min_x + eps >= corner[0]]
        max_x_corners = [corner for corner in corners if max_x - eps <= corner[0]]
        min_y_corners = [corner for corner in corners if min_y + eps >= corner[1]]
        max_y_corners = [corner for corner in corners if max_y - eps <= corner[1]]

        top_left = min(min_x_corners, key = lambda corner: corner[1])
        top_right = max(min_y_corners, key = lambda corner: corner[0])
        bottom_left = min(max_y_corners, key = lambda corner: corner[0])
        bottom_right = max(max_x_corners, key = lambda corner: corner[1])

        x = [top_left[0], top_right[0], bottom_right[0], bottom_left[0]]
        y = [top_left[1], top_right[1], bottom_right[1], bottom_left[1]]
        return x, y

    # Store slope, y-intercept info for each detected line
    m_b_pairs = set()
    for line in lines:
        x1,y1,x2,y2 = line[0]
        if x1 != x2:
            slope = (y2 - y1) / (x2 - x1)
            b = -slope * x1 + y1
        else:
            slope = np.inf
            b = x1
        m_b_pairs.add((slope, b - (b % 4)))
        add_line_to_img(img, slope, b - (b % 4))

    m_b_pairs = list(m_b_pairs)

    # Find intersections for lines which intersect at least 4 others
    # perpendicularly
    corners = set()
    for i in range(len(m_b_pairs)):
        m1, b1 = m_b_pairs[i]
        perpendicular_lines = []
        for j in range(i, len(m_b_pairs)):
            m2, b2 = m_b_pairs[j]
            if are_perp(m1, m2):
                perpendicular_lines.append((m2, b2))

        if (len(perpendicular_lines) >= 3):
            for m2, b2 in perpendicular_lines:
                x, y = solve_for_intersections(m1, b1, m2, b2)
                corners.add((int(x), int(y)))

    x, y = find_corners(list(corners))
    if plot_intermediate_results:
        plt.figure()
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.plot(x + x[:1], y + y[:1], 'r')
        for corner in corners:
            plt.plot(corner[0], corner[1], 'r*')
        plt.title("Hough " + img_name)
        plt.show()

    return x, y

# Set up and compute the homography transformation

# 4 pixels per mm
f = 4
rubiks_mm = 57

# Dimensions of the face are 57 mm x 57 mm
x2 = [0, rubiks_mm*f, rubiks_mm*f, 0]
y2 = [0, 0, rubiks_mm*f, rubiks_mm*f]

def compute_homography(img_name, x, y):
    img = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)

    src = np.vstack((y, x)).T
    dst = np.vstack((x2, y2)).T
    tform = tf.estimate_transform('projective', src, dst)

    return tf.warp(img, inverse_map=tform.inverse, output_shape=(rubiks_mm * f, rubiks_mm * f))

# Find the mean RGB color for each square
def get_face_colors(transformed_face):
    avg_sqr_colors = []
    for i in range(3):
        for j in range(3):
            top_left = (i, j)
            bottom_right = (i+1, j+1)
            block_colors = []
            for x in range(int(f * rubiks_mm * top_left[0]/3), int(f * rubiks_mm * bottom_right[0]/3)):
                for y in range(int(f * rubiks_mm * top_left[1]/3), int(f * rubiks_mm * bottom_right[1]/3)):
                    block_colors.append(transformed_face[y, x])
            avg_sqr_colors.append(255 * np.mean(block_colors, axis=0))

    return avg_sqr_colors

# Generate and plot the cube layout
def plot_cube_layout(all_colors, with_spaces=True):
    straight_img = np.zeros((18, 3, 3))
    for i in range(54):
        straight_img[int(i / 3), i % 3, 0] = int(all_colors[i][0])
        straight_img[int(i / 3), i % 3, 1] = int(all_colors[i][1])
        straight_img[int(i / 3), i % 3, 2] = int(all_colors[i][2])

    if with_spaces:
        s = 35
        ss = 3*s + 3 + 1
        def create_side_model(side_colors):
            side = np.zeros((ss, ss, 3))
            for i in range(3):
                for j in range(3):
                    side[(s+1)*i+1:(s+1)*i+(s+1), (s+1)*j+1:(s+1)*j+(s+1)] = side_colors[3*i + j]

            return side

        show_img = np.zeros((9*s + 9 + 1, 12*s + 12 + 1, 3))

        show_img[0:ss, ss-1:2*ss-1] = create_side_model(all_colors[0:9])
        show_img[ss-1:2*ss-1, 0:ss] = create_side_model(all_colors[9:18])
        show_img[ss-1:2*ss-1, ss-1:2*ss-1] = create_side_model(all_colors[18:27])
        show_img[ss-1:2*ss-1, 2*ss-2:3*ss-2] = create_side_model(all_colors[27:36])
        show_img[ss-1:2*ss-1, 3*ss-3:4*ss-3] = create_side_model(all_colors[36:45])
        show_img[2*ss-2:3*ss-2, ss-1:2*ss-1] = create_side_model(all_colors[45:54])

    else:
        show_img = np.zeros((9, 12, 3))

        show_img[0:3, 3:6] = straight_img[:3, :3]
        show_img[3:6, 0:3] = straight_img[3:6, :3]
        show_img[3:6, 3:6] = straight_img[6:9, :3]
        show_img[3:6, 6:9] = straight_img[9:12, :3]
        show_img[3:6, 9:12] = straight_img[12:15, :3]
        show_img[6:9, 3:6] = straight_img[15:18, :3]

    plt.figure()
    plt.imshow(np.ndarray.astype(show_img, np.uint8))
    plt.title("Cube Layout")
    plt.show()

# Print the instructions step-by-step to solve the cube
def print_solution_instructions(solution, face_center_labels):
    print("=" * 10)
    print("SOLUTION")
    print("=" * 10)
    for move in solution:
        f = move[0]

        # Find the top label
        if f == 'L' or f == 'F' or f == 'R' or f == 'B':
            top_label = face_center_labels[0]
        elif f == 'U':
            top_label = face_center_labels[4]
        else:
            top_label = face_center_labels[2]

        # Find the front label
        if f == 'U': front_label = face_center_labels[0]
        elif f == 'L': front_label = face_center_labels[1]
        elif f == 'F': front_label = face_center_labels[2]
        elif f == 'R': front_label = face_center_labels[3]
        elif f == 'B': front_label = face_center_labels[4]
        else: front_label = face_center_labels[5]

        # Find the direction to move
        if len(move) == 1:
            direction = "clockwise"
        elif move[1] == '\'':
            direction = "counterclockwise"
        else:
            direction = "clockwise twice"

        # Print the instructions
        print("Orient the cube so that center " + str(front_label) + " is facing you and center " + str(top_label) + " is on top")
        print("Turn the side of the cube facing you " + direction)
        input("Press the ENTER key to show the next step")
        print("=" * 10)

    print("Congratulations, you just solved the cube!")


# Find the average RGB value for each square on the cube
all_colors = []
for side in sides:
    img_name = "test_images/" + side + dataset + ".jpg"
    y, x = get_corners_hough(img_name)
    warped = compute_homography(img_name, x, y)
    all_colors += get_face_colors(warped)

    # Plot the transformed face
    if plot_intermediate_results:
        plt.figure()
        plt.imshow(warped)
        plt.plot(x2 + x2[:1], y2 + y2[:1], 'r-')
        plt.plot([19*f, 19*f], [1, 57*f], 'r-')
        plt.plot([38*f, 38*f], [1, 57*f], 'r-')
        plt.plot([1, 57*f], [19*f, 19*f], 'r-')
        plt.plot([1, 57*f], [38*f, 38*f], 'r-')
        plt.title("Transformed " + img_name)
        plt.show()

# Performs k-means clustering on the RGB values and give them
# each one of k labels, where k = 6
centers, _ = kmeans(all_colors, k_or_guess = 6)
labels = []
for color in all_colors:
    dists = []
    for center in centers:
        dists.append(np.linalg.norm(color - center))
    labels.append(np.argmin(dists))

# Plot the Rubik's cube's faces using the cluster center as the
# representative color for each square
all_centers = [centers[label] for label in labels]
plot_cube_layout(all_centers)

# Assign a color to each label, consistent with the standard
# Rubik's cube color conventions
labels_to_colors = [""] * 6
labels_to_colors[labels[4]]  = "y"
labels_to_colors[labels[13]] = "b"
labels_to_colors[labels[22]] = "r"
labels_to_colors[labels[31]] = "g"
labels_to_colors[labels[40]] = "o"
labels_to_colors[labels[49]] = "w"

# Create the input string for the solver algorithm using the
# label to color mapping created previously
input_str = ""
for label in labels:
    input_str += labels_to_colors[label]

# Call the solving algorithm
solution = utils.solve(input_str, 'Kociemba')

# Plot reference diagram
plt.figure()
ref = np.zeros((6, 1, 3))
for i in range(len(centers)):
    ref[i] = centers[i]
plt.imshow(np.ndarray.astype(ref, np.uint8))
plt.title("Reference Diagram for Solution")
plt.show(block=False)

# Output the solution step-by-step
face_center_labels = [labels[9*i+4] for i in range(6)]
print(solution)
solution_strs = [str(move) for move in solution]
print_solution_instructions(solution_strs, face_center_labels)
