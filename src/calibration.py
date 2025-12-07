import cv2
import numpy as np
import glob

# ---- SETTINGS ----
images_path = "src/calibration/*.jpg"
pattern_size = (9, 6)       
square_size = 0.025         # in meters

# -------------------

# Prepare object points like:
# (0,0,0), (1,0,0), (2,0,0) ... scaled by square_size
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D points
imgpoints = []  # 2D points

images = glob.glob(images_path)

print(f"Found {len(images)} images")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find corners
    found, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if found and 'capture' in fname:
    # if found:
        #display conrners
        cv2.drawChessboardCorners(img, pattern_size, corners, found)
        cv2.imwrite(fname.replace("calibration\\", "calibration/annotated/"), img)
        
        print("✔ Corners found in", fname.replace("calibration\\", "calibration/annotated/"))

        # Refine corners
        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001),
        )

        imgpoints.append(corners2)
        objpoints.append(objp)
    else:
        print("✖ Corners NOT found in", fname)

print(len(objpoints), "valid images for calibration")
# Run calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("\n=== CALIBRATION COMPLETE ===")
print("Camera Matrix:\n", camera_matrix)
print("\nDistortion Coefficients:\n", dist_coeffs)

# Save results
np.save("./src/camera_matrix.npy", camera_matrix)
np.save("./src/dist_coeffs.npy", dist_coeffs)

print("\nSaved as camera_matrix.npy and dist_coeffs.npy")
print(ret, "re-projection error")