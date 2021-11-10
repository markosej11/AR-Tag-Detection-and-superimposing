# Libraries used

import cv2 as cv
import numpy as np

# Importing Video files and required Images as per user input

print("Enter the video you want to play (1/ 2/ 3)")
print("1. Tag0")
print("2. Tag1")
print("3. Tag2")
print("4. Multiple tags")
choice = input()
video = None
if choice == '1':
    video = cv.VideoCapture('VideoDataset/Tag0.mp4')
elif choice == '2':
    video = cv.VideoCapture('VideoDataset/Tag1.mp4')
elif choice == '3':
    video = cv.VideoCapture('VideoDataset/Tag2.mp4')
elif choice == '4':
    video = cv.VideoCapture('VideoDataset/multipleTags.mp4')
else:
    print("Invalid input!! ")
    exit(0)


pic = cv.imread('ReferenceImages/Lena.png')               
(pic_x, pic_y, ch) = pic.shape
image_x_points, image_y_points = 400, 400

# Cube coordinates for drawing 3D cube

cube_coordinates = np.float32([[0, 0, 0], [0, 500, 0], [500, 500, 0],
                               [500, 0, 0], [0, 0, -200], [0, 500, -200],
                               [500, 500, -200], [500, 0, -200]])  

# Function for calculating the H matrix

def homographymatrix(position, cam_x, cam_y):
    
    # World coordinates 
    
    if position == 'bR':
        xw1, yw1 = coordinates[0][0][0], coordinates[0][0][1]
        xw2, yw2 = coordinates[1][0][0], coordinates[1][0][1]
        xw3, yw3 = coordinates[2][0][0], coordinates[2][0][1]
        xw4, yw4 = coordinates[3][0][0], coordinates[3][0][1]
    elif position == 'bL':
        xw1, yw1 = coordinates[1][0][0], coordinates[1][0][1]
        xw2, yw2 = coordinates[2][0][0], coordinates[2][0][1]
        xw3, yw3 = coordinates[3][0][0], coordinates[3][0][1]
        xw4, yw4 = coordinates[0][0][0], coordinates[0][0][1]
    elif position == 'tL':
        xw1, yw1 = coordinates[2][0][0], coordinates[2][0][1]
        xw2, yw2 = coordinates[3][0][0], coordinates[3][0][1]
        xw3, yw3 = coordinates[0][0][0], coordinates[0][0][1]
        xw4, yw4 = coordinates[1][0][0], coordinates[1][0][1]
    elif position == 'tR':
        xw1, yw1 = coordinates[3][0][0], coordinates[3][0][1]
        xw2, yw2 = coordinates[0][0][0], coordinates[0][0][1]
        xw3, yw3 = coordinates[1][0][0], coordinates[1][0][1]
        xw4, yw4 = coordinates[2][0][0], coordinates[2][0][1]
    else:
        raise ValueError("Invalid corner provided!")

    
    # Camera coordinates 
    
    xc1, yc1 = 0, 0
    xc2, yc2 = cam_x, 0
    xc3, yc3 = cam_x, cam_y
    xc4, yc4 = 0, cam_y

    
    # Calculating the A Matrix
    
    a = [[xw1, yw1, 1, 0, 0, 0, -xc1*xw1, -xc1*yw1, -xc1],
         [0, 0, 0, xw1, yw1, 1, -yc1*xw1, -yc1*yw1, -yc1],
         [xw2, yw2, 1, 0, 0, 0, -xc2*xw2, -xc2*yw2, -xc2],
         [0, 0, 0, xw2, yw2, 1, -yc2*xw2, -yc2*yw2, -yc2],
         [xw3, yw3, 1, 0, 0, 0, -xc3*xw3, -xc3*yw3, -xc3],
         [0, 0, 0, xw3, yw3, 1, -yc3*xw3, -yc3*yw3, -yc3],
         [xw4, yw4, 1, 0, 0, 0, -xc4*xw4, -xc4*yw4, -xc4],
         [0, 0, 0, xw4, yw4, 1, -yc4*xw4, -yc4*yw4, -yc4]]

    # Using SVD to to find the homography

    u, s, vt = np.linalg.svd(a, full_matrices=True)
    h = np.array(vt[8, :]/vt[8, 8]).reshape((3, 3))

    h_inv = np.linalg.inv(h)
    return h, h_inv

    # Wrap function to ........ 

def warp_map(yc, xc, h_mat, img):
    hh = img.shape[0]
    ww = img.shape[1]
    new_coord = np.matmul(h_mat, np.array([[xc], [yc], [1.0]]))
    new_i = int(new_coord[1][0] / new_coord[2][0])
    new_j = int(new_coord[0][0] / new_coord[2][0])
    if 0 <= new_i < hh and 0 <= new_j < ww:
        return img[new_i, new_j]
    return [0, 0, 0]


def warp_backward(img, h_mat, coord):
    yy, xx = coord
    new_image = np.array([[warp_map(ii, jj, h_mat, img) for jj in range(0, xx, 12)] for ii in range(0, yy, 12)])
    return new_image


def warp_forward(img, h_mat, coord, corners):
    new_image = np.zeros((coord[0], coord[1], 3), dtype=np.uint8)
    new_image = cv.drawContours(new_image, [corners], 0, (255, 255, 255), thickness=-1)
    min_w = np.min([c[0][0] for c in corners])
    min_h = np.min([c[0][1] for c in corners])
    max_w = np.max([c[0][0] for c in corners])
    max_h = np.max([c[0][1] for c in corners])
    vect_func = np.vectorize(warp_map, otypes=[np.ndarray], excluded=[2, 3, "h_mat", "img"], cache=True)

    small_image = vect_func(np.array([[yv for xv in range(min_w, max_w)] for yv in range(min_h, max_h)]),
                            np.array([[xv for xv in range(min_w, max_w)] for yv in range(min_h, max_h)]),
                            h_mat=h_mat, img=img)

    new_image[min_h:max_h, min_w:max_w] = np.array([[a for a in row] for row in small_image], dtype=np.uint8)
    return new_image


def is_paper(image):

    # Check if the detected 4-corner contour is an actual tag or the paper by checking if the border is white
    
    border_image = cv.rectangle(image, (10, 10), (70, 70), 0, cv.FILLED)
    border_portion = np.sum(border_image) / ((80 * 80) * ((64 - 36) / 64)) / 255
    return border_portion > 0.92



# Function to find the orientation of the tag i.e the location of the white cell

def orientation(img_data):

    # Finding the average intensity in each of the four corner cells

    corners = [("tL", np.sum(img_data[100:150, 100:150]) / 2500),
               ("tR", np.sum(img_data[100:150, 250:300]) / 2500),
               ("bR", np.sum(img_data[250:300, 250:300]) / 2500),
               ("bL", np.sum(img_data[250:300, 100:150]) / 2500)]
    corners.sort(key=lambda tup: tup[1], reverse=True)
    return corners[0][0]

# Function to get the tag ID

def get_id(input_image):
    keys = ['TOPL', 'TOPR', 'BOTR', 'BOTL']
    position = {'BOTL': [200, 250, 150, 200],
                'BOTR': [200, 250, 200, 250],
                'TOPR': [150, 200, 200, 250],
                'TOPL': [150, 200, 150, 200]}
    
    # Looping through each of the 4 cells in the center of the AR tag and finding the average intensity in each
    
    tag_id = "".join("1" if np.sum(input_image[position[keys[o]][0]:position[keys[o]][1],
                                               position[keys[o]][2]:position[keys[o]][3]]) / 2500 > 216 else "0"
                     for o in range(0, 4))
    return tag_id

# Function to get the rotational matrix and the transverse vector

def get_matrix_cube(inv_h):
    k_mat = np.array([[1406.08415449821, 0, 0],
                      [2.20679787308599, 1417.99930662800, 0],
                      [1014.13643417416, 566.347754321696, 1]]).T
    inv_k_mat = np.linalg.inv(k_mat)
    b_mat = np.matmul(inv_k_mat, inv_h)
    b1 = b_mat[:, 0].reshape(3, 1)
    b2 = b_mat[:, 1].reshape(3, 1)
    r3 = np.cross(b_mat[:, 0], b_mat[:, 1])
    b3 = b_mat[:, 2].reshape(3, 1)
    scalar = 2/(np.linalg.norm(inv_k_mat.dot(b1))+np.linalg.norm(inv_k_mat.dot(b2)))
    t_ = scalar*b3
    r1 = scalar*b1
    r2 = scalar*b2
    r3 = (r3 * scalar * scalar).reshape(3, 1)
    r_mat = np.concatenate((r1, r2, r3), axis=1)
    return r_mat, t_, k_mat


# Function to draw the three dimensional figure from the coordinates of the cube


def draw3d(frame, points_3d):
    points_3d = np.int32(points_3d).reshape(-1, 2)
    frame = cv.drawContours(frame, [points_3d[:4]], -1, (0, 255, 0), 3)   # Ground plane
    for i_, j_ in zip(range(4), range(4, 8)):                                    # Z Axis planes
        frame = cv.line(frame, tuple(points_3d[i_]), tuple(points_3d[j_]), (0, 0, 255), 3)
    frame = cv.drawContours(frame, [points_3d[4:]], -1, (255, 0, 0), 3)   # Top plane
    return frame

# For writing each frame to make the final video

scale = 0.4
frame_width = int(video.get(3) * scale * 2)
frame_height = int(video.get(4) * scale * 2)
out1 = cv.VideoWriter('out1py.mp4', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (frame_width, frame_height))

# Main function

while True:
    # Extracting Video Frame Size
    ret, main_frame = video.read()
    if ret is None or ret is False:
        break
    mf_original = main_frame.copy()
     # Converting image into grayscale and using contour function
    rows, cols, chs = main_frame.shape
    mf_gray = cv.cvtColor(main_frame, cv.COLOR_BGR2GRAY)
    mf_thresh = cv.threshold(mf_gray, 200, 255, 0)[1]
    contours, heirarchy = cv.findContours(mf_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    ThreeDCoordinates = main_frame
    Lena_result = main_frame
    tag_upright = np.zeros((image_y_points, image_x_points), dtype=np.uint8)
    warpped_image = np.zeros((main_frame.shape[0], main_frame.shape[1], 3), dtype=np.uint8)
    for contour in contours:
        perimeter = cv.arcLength(contour, True)
        coordinates = cv.approxPolyDP(contour, 0.01 * perimeter, True)
        area = cv.contourArea(contour)
        # shortlisting Contour candidates with area approximately equal to the Tag area among the multiple cntours detected.
        if 600 <= area < 24000:    
            # Filtering Contours with 4 corners
            if len(coordinates) == 4:  
                try:
                    H_video, H_inv_video = homographymatrix('bR', image_x_points, image_y_points)
                except np.linalg.LinAlgError:
                    continue
                               
                Video_Unwarp = cv.resize(warp_backward(main_frame, H_inv_video, (image_y_points, image_x_points)),
                                         dsize=None, fx=12, fy=12)
                Video_Unwarp_Gray = cv.cvtColor(Video_Unwarp, cv.COLOR_BGR2GRAY)
                Video_Unwarp_Thresh = cv.threshold(Video_Unwarp_Gray, 200, 255, cv.THRESH_BINARY)[1]
                ratio_image = cv.resize(Video_Unwarp_Thresh, dsize=None, fx=0.2, fy=0.2)
                if not is_paper(ratio_image):
                    # Computing the position of the video using thresholded unwarpped video image
                    positionID = orientation(Video_Unwarp_Thresh)
                    Font = cv.FONT_HERSHEY_DUPLEX
                    # Computing the video GetID of the thresholded unwarpped video image (This is done only once)
                    upright_dict = {"tL": cv.ROTATE_180,
                                    "tR": cv.ROTATE_90_CLOCKWISE,
                                    "bL": cv.ROTATE_90_COUNTERCLOCKWISE}
                    tag_upright = np.copy(Video_Unwarp_Thresh) if positionID == "bR" \
                        else cv.rotate(Video_Unwarp_Thresh, upright_dict[positionID])
                    Video_ID = int(get_id(tag_upright))
                    x = []
                    z = ""
                    res = [int(x) for x in str(Video_ID)]
                    for i in range(0, len(res)):
                        res[i] = int(res[i])
                    for j in range(len(res)):
                        x.append((res[j])*pow(2, j))
                    for k in range(len(x)):
                        z = z + str(x[k])
                    tx1, ty1 = coordinates[0][0][0], coordinates[0][0][1]
                    cv.putText(mf_original, "Tag ID: %s" % z, (tx1-50, ty1-50), Font, 1, (0, 0, 225), 2, cv.LINE_AA)
                    cv.putText(tag_upright, "Tag ID: %s" % z, (50, 50), Font, 0.75, 160, 2, cv.LINE_AA)
                    # Recalculating H and H inverse using the position information obtained
                    try:
                        H_picture, H_Inverse_picture = homographymatrix(positionID, pic_x, pic_y)
                    except np.linalg.LinAlgError:
                        continue
                    # Preparing the image and projecting into the video using the recalculated Homography matrix
                    warpped_image = np.bitwise_or(warpped_image,
                                                  warp_forward(pic, H_picture, (rows, cols), coordinates))
                    warpped_image_grayscale = cv.cvtColor(warpped_image, cv.COLOR_BGR2GRAY)
                    # Creating Mask and merging the key frame with the warpped
                    # image using bitwise_and and add opencv functions
                    warpped_image_threshold = cv.threshold(warpped_image_grayscale, 0, 250, cv.THRESH_BINARY_INV)[1]
                    mf_slotted = cv.bitwise_and(mf_original, mf_original, mask=warpped_image_threshold)
                    Lena_result = cv.add(mf_slotted, warpped_image)
                    # Computing the Projection Matrix for placing 3D objects
                    R_mat, t, K_mat = get_matrix_cube(H_Inverse_picture)
                    ThreeDpoints, jacobian = cv.projectPoints(cube_coordinates, R_mat, t, K_mat, np.zeros((1, 4)))
                    ThreeDCoordinates = draw3d(main_frame, ThreeDpoints)
    
    # Display the Original Keyframe and Final Result
    mf_original_display = cv.resize(mf_original, dsize=None, fx=scale, fy=scale)
    Lena_result_display = cv.resize(Lena_result, dsize=None, fx=scale, fy=scale)
    ThreeDCoordinates_display = cv.resize(ThreeDCoordinates, dsize=None, fx=scale, fy=scale)
    hpad = (mf_original_display.shape[1] - tag_upright.shape[1]) // 2
    vpad = (mf_original_display.shape[0] - tag_upright.shape[0]) // 2
    show_three = True
    if show_three:
        ThreeDCoordinates_display = cv.copyMakeBorder(ThreeDCoordinates_display,
                                                      top=0, bottom=0,
                                                      left=ThreeDCoordinates_display.shape[1] // 2,
                                                      right=ThreeDCoordinates_display.shape[1] // 2,
                                                      borderType=cv.BORDER_CONSTANT, value=(48, 48, 48))
        image_display = np.concatenate((np.concatenate((mf_original_display, Lena_result_display), axis=1),
                                        ThreeDCoordinates_display), axis=0)
    else:
        extra_image = cv.copyMakeBorder(tag_upright, top=vpad, bottom=vpad, left=hpad, right=hpad,
                                        borderType=cv.BORDER_CONSTANT, value=128)
        extra_image = cv.cvtColor(extra_image, cv.COLOR_GRAY2RGB)
        image_display = np.concatenate((np.concatenate((mf_original_display, Lena_result_display), axis=1),
                                        np.concatenate((ThreeDCoordinates_display, extra_image), axis=1)),
                                       axis=0)
    out1.write(image_display)
    cv.imshow("Result", image_display)
    key = cv.waitKey(1)
    if key == 27:
        break

video.release()
# cv.waitKey(0)
cv.destroyAllWindows()
