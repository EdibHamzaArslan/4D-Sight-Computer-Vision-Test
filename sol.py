#---------------------
#| 21th September 2021
#| Edib Hamza Arslan
#|
#---------------------

import cv2 
import numpy as np 
import argparse



data2d_path = 'vr2d.npy'
data3d_path = 'vr3d.npy'


def plot_trajectory(input_path, output_path):
	''' In this function, calculating project points and 
	plotting the trajectory 
	Inputs:
		input_path - str
		output_path - str
	Outputs:
		None
	'''
	
	img = cv2.imread(input_path)

	dist_coeffs = np.zeros((5,1)) # data points have extra dimension

	camera_matrix = np.eye(3)
	camera_matrix[0][0] = 100
	camera_matrix[1][1] = 100
	camera_matrix[0][2] = 960
	camera_matrix[1][2] = 540

	# Loading the points
	points_2d = np.load(data2d_path)
	points_3d = np.load(data3d_path)


	# Trajectory points
	points = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).reshape(-1, 3)

	# calculate rotation and translation vectors
	_, rotation_vector, translation_vector = cv2.solvePnP(points_3d, 
														  points_2d, 
													      camera_matrix, 
														  dist_coeffs, 
														  flags=0)
	
	img_points, jacobian = cv2.projectPoints(points,
											 rotation_vector, 
											 translation_vector, 
											 camera_matrix, 
											 dist_coeffs)



	x1 = tuple([int(x) for x in img_points[3].ravel()])

	y1 = tuple([int(x) for x in img_points[0].ravel()])
	y2 = tuple([int(x) for x in img_points[1].ravel()])
	y3 = tuple([int(x) for x in img_points[2].ravel()])


	cv2.line(img, x1, y1, (255,0,0), 10)
	cv2.line(img, x1, y2, (0,255,0), 10)
	cv2.line(img, x1, y3, (0,0,255), 10)

	# Save output
	cv2.imwrite(output_path, img)


ap = argparse.ArgumentParser()

ap.add_argument("-i", "--input", type=str, required=True,
                help="input path")

ap.add_argument("-o", "--output", type=str, required=True,
                help="output path")

args = vars(ap.parse_args())


plot_trajectory(args['input'], args['output'])



