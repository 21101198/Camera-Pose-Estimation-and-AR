import cv2
import numpy as np
import glob

# 체스보드 내부 코너 수
pattern_size = (9, 6)
square_size = 1.0

# 3D 좌표 준비
objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D 점들
imgpoints = []  # 2D 점들

# 체스보드 이미지 불러오기 (예: 'calib_images/img1.jpg' 등)
images = glob.glob('calib_images/*.jpg')  # 📁 이미지 폴더 위치 맞게 수정!

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray, pattern_size)

    if found:
        objpoints.append(objp)
        corners_sub = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners_sub)

        cv2.drawChessboardCorners(img, pattern_size, corners_sub, found)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# 캘리브레이션 실행
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 결과 저장
np.savez('calibration_data.npz', cameraMatrix=camera_matrix, distCoeffs=dist_coeffs)

print("캘리브레이션 완료. calibration_data.npz 저장됨.")
