import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

from pose_vector_to_transformation_matrix import \
    pose_vector_to_transformation_matrix
from project_points import project_points
from undistort_image import undistort_image
from undistort_image_vectorized import undistort_image_vectorized

K_path = r"C:\Users\Pham Quyen\PycharmProjects\pythonProject\K.txt"
D_path = r"C:\Users\Pham Quyen\PycharmProjects\pythonProject\D.txt"
poses_path = r"C:\Users\Pham Quyen\PycharmProjects\pythonProject\poses.txt"
def main():
    pass

    # load camera poses

    # each row i of matrix 'poses' contains the transformations that transforms
    # points expressed in the world frame to
    # points expressed in the camera frame

def compute_transformation(row):

    omega = row[:3]  # Vector góc quay
    t = row[3:]      # Vector tịnh tiến
    theta = np.linalg.norm(omega)  # Độ lớn của vector góc quay

    if theta != 0:
        k = omega / theta  # Vector đơn vị của trục quay
    else:
        k = np.array([0, 0, 0])  # Nếu không có góc quay

    # Tạo ma trận [k]
    kx, ky, kz = k
    K = np.array([
        [0, -kz, ky],
        [kz, 0, -kx],
        [-ky, kx, 0]
    ])

    # Tính ma trận quay R theo công thức Rodrigues
    I = np.eye(3)
    R = I + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

    # Tạo ma trận 4x4
    T_3x4 = np.hstack((R, t.reshape(3, 1)))

    return T_3x4  # Trả về ma trận 3x4

    return T

def load_camera_poses(file_path):
    """
    Đọc file poses.txt và chuyển đổi thành danh sách các ma trận biến đổi 4x4.


    """
    poses = np.loadtxt(poses_path)  # Đọc file
    num_poses = poses.shape[0]  # Số dòng dữ liệu

    pose_matrices = [compute_transformation(poses[i, :]) for i in range(num_poses)]
    return np.array(pose_matrices)  # Trả về danh sách các ma trận 4x4

# Đọc file poses.txt và hiển thị kết quả

camera_poses = load_camera_poses(poses_path)

print("Loaded {} camera poses".format(len(camera_poses)))
print("First pose matrix:\n", camera_poses[0])
img_gray= cv2.imread(r"C:\Users\Pham Quyen\Downloads\Exercise_01\Exercise_01\data\images_undistorted\img_0001.jpg")
    # TODO: Your code here
K=np.loadtxt(K_path)
print(K)
    # define 3D corner positions
    # [Nx3] matrix containing the corners of the checkerboard as 3D points
    # (X,Y,Z), expressed in the world coordinate system

    # TODO: Your code here
size_x = np.linspace(-1, 9, 11)*0.04
size_y = np.linspace(-1, 6, 8)*0.04
points_x, points_y = np.meshgrid(size_x, size_y)

# Chuyển thành dạng tọa độ đồng nhất (X_W, Y_W, Z_W=0, 1)
points_W = np.vstack([points_x.ravel(), points_y.ravel(), np.zeros(points_x.size), np.ones(points_x.size)])
print(points_W)
point_list = [points_W[:, i].reshape(4, 1) for i in range(points_W.shape[1])]
print(point_list[5])
# Chuyển đổi thành mảng Nx3 với Z = 0
image_points = []
save_point =[]
# Duyệt từng điểm để tính toán tọa độ ảnh
for point in point_list:
    P_C = camera_poses[0] @ point  # (3x4) x (4x1) → (3x1)
    P_img = K @ P_C            # (3x3) x (3x1) → (3x1)
    P_img /= P_img[2]          # Chuẩn hóa tọa độ ảnh
    image_points.append(P_img[:2].flatten())  # Lưu u, v


# Chuyển danh sách sang numpy (2xN)
image_points = np.array(image_points).T

# Làm tròn tọa độ ảnh
pixel_coords = np.round(image_points).astype(int)

# Tạo ảnh xám để vẽ lên


# Vẽ từng điểm ảnh
for i in range(pixel_coords.shape[1]):
   cv2.circle(img_gray, tuple(pixel_coords[:, i]), 5, (0, 0, 255), -1)
   save_point.append(tuple(pixel_coords[:, i]))
point_hcn=[]
def update_z(point, new_z):
    updated_point = point.copy()  # Tạo bản sao để không làm thay đổi danh sách gốc
    updated_point[2, 0] = new_z   # Cập nhật giá trị Z
    return updated_point
updated_points = {
    27: update_z(point_list[27], -0.08),
    29: update_z(point_list[29], -0.08),
    49: update_z(point_list[49], -0.08),
    51: update_z(point_list[51], -0.08),
    28: update_z(point_list[27], 0),
    30:update_z(point_list[29], 0),
    50:update_z(point_list[49], 0),
    52:update_z(point_list[51], 0)
}
image_points = []  # Danh sách lưu tọa độ ảnh

for i in [27, 29, 49, 51,28,30,50,52]:
    P_C = camera_poses[0] @ updated_points[i]  # Chuyển sang hệ tọa độ camera
    P_img = K @ P_C   # Chuyển đổi sang hệ tọa độ ảnh
    P_img /= P_img[2]  # Chuẩn hóa tọa độ ảnh
    image_points.append(P_img[:2].flatten())  # Lưu tọa độ ảnh (u, v)

# Chuyển sang numpy array (2x4) để dễ sử dụng
image_points = np.array(image_points).T

# Làm tròn tọa độ ảnh để sử dụng trên hình
pixel_coords = np.round(image_points).astype(int)
pixel_coords = pixel_coords.T
cv2.line(img_gray, tuple(pixel_coords[0]), tuple(pixel_coords[1]), (0, 0, 255), 3)
cv2.line(img_gray, tuple(pixel_coords[0]), tuple(pixel_coords[2]), (0, 0, 255), 3)
cv2.line(img_gray, tuple(pixel_coords[1]), tuple(pixel_coords[3]), (0, 0, 255), 3)
cv2.line(img_gray, tuple(pixel_coords[2]), tuple(pixel_coords[3]), (0, 0, 255), 3)
cv2.circle(img_gray, tuple(pixel_coords[0]), 5, (0, 0, 255), -1)
cv2.circle(img_gray, tuple(pixel_coords[1]), 5, (0, 0, 255), -1)
cv2.circle(img_gray, tuple(pixel_coords[2]), 5, (0, 0, 255), -1)
cv2.circle(img_gray, tuple(pixel_coords[3]), 5, (0, 0, 255), -1)
cv2.circle(img_gray, tuple(pixel_coords[4]), 5, (0, 0, 255), -1)
cv2.circle(img_gray, tuple(pixel_coords[5]), 5, (0, 0, 255), -1)
cv2.circle(img_gray, tuple(pixel_coords[6]), 5, (0, 0, 255), -1)
cv2.circle(img_gray, tuple(pixel_coords[7]), 5, (0, 0, 255), -1)

def convert_to_xyz(homogeneous_point):
    return homogeneous_point[:3] / homogeneous_point[3]
convert_to_xyz(point_list[27])
convert_to_xyz(point_list[29])
convert_to_xyz(point_list[49])
convert_to_xyz(point_list[51])
cv2.line(img_gray, tuple(pixel_coords[0]),save_point[27], (0, 0, 255), 3)
cv2.line(img_gray, tuple(pixel_coords[1]),save_point[29], (0, 0, 255), 3)
cv2.line(img_gray, tuple(pixel_coords[2]),save_point[49], (0, 0, 255), 3)
cv2.line(img_gray, tuple(pixel_coords[3]),save_point[51], (0, 0, 255), 3)
cv2.line(img_gray,save_point[27],save_point[29],(0,0,255),3)
cv2.line(img_gray,save_point[27],save_point[49],(0,0,255),3)
cv2.line(img_gray,save_point[29],save_point[51],(0,0,255),3)
cv2.line(img_gray,save_point[49],save_point[51],(0,0,255),3)
# Hiển thị hình ảnh

cv2.imshow("Image", img_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Chờ nhấn phím Space (phím cách) để tắt


while True:
    key = cv2.waitKey(1) & 0xFF  # Đọc phím bấm
    if key == 32:  # 32 là mã ASCII của phím Space
        break

plt.clf()
plt.close()
plt.imshow(img_undistorted, cmap='gray')

lw = 3

# base layer of the cube



    # TODO: Your code here

    # load one image with a given index
    # TODO: Your code here


    # project the corners on the image
    # compute the 4x4 homogeneous transformation matrix that maps points
    # from the world to the camera coordinate frame

    # TODO: Your code here


    # transform 3d points from world to current camera pose
    # TODO: Your code here

    # undistort image with bilinear interpolation


if __name__ == "__main__":
    main()
