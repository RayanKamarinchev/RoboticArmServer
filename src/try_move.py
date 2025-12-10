camera_coords = [0.49574684, 0.0616682, 0.42840817]
desired_coords = [0.2, camera_coords[1], 0.05]
degrees = get_move_angles(camera_coords, desired_coords, [alpha, beta, gamma])
alpha, beta, gamma = degrees
print(a * np.sin(alpha) - b * np.sin(alpha + beta) + c * np.sin(alpha + beta + gamma) + baseElevation)
print(a * np.cos(alpha) - b * np.cos(alpha + beta) + c * np.cos(alpha + beta + gamma))
print("Final Angles:", [np.degrees(alpha), np.degrees(beta), np.degrees(gamma)])