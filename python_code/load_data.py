import numpy as np

# Đường dẫn tuyệt đối của file
K_path = r"C:\Users\Pham Quyen\PycharmProjects\pythonProject\K.txt"
D_path = r"C:\Users\Pham Quyen\PycharmProjects\pythonProject\D.txt"
poses_path = r"C:\Users\Pham Quyen\PycharmProjects\pythonProject\poses.txt"

# Đọc file bằng đường dẫn tuyệt đối
K = np.loadtxt(K_path)
D = np.loadtxt(D_path).reshape(-1, 1)  # Chuyển D thành vector cột
poses = np.loadtxt(poses_path)

print("Camera matrix K:\n", K)
print("Distortion coefficients D:\n", D)
print("Loaded {} camera poses".format(poses.shape[0]))
