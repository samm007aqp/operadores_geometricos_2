import cv2
import numpy as np 
from matplotlib import pyplot as plt 
import math as ma 
from decimal import Decimal, ROUND_HALF_UP


def size_angle(teta,h,w):
	if teta< ma.pi/2:
		new_w = ((w)*ma.cos(teta)+( (h)*ma.sin(teta)))
		new_h = ((w)*ma.sin(teta)+( (h)*ma.cos(teta)))
	else :
		h_ = w
		w_ = h
		teta = teta-ma.pi/2
		new_w = ( (w_)*ma.cos(teta)+( (h_)*ma.sin(teta)))
		new_h = ( (w_)*ma.sin(teta)+( (h_)*ma.cos(teta)))
	return abs(int(new_h)), abs(int(new_w))


def rotacion(input,matrix,B,rows,cols):  ## rotacion con el metodo cv2.solve()
	X = np.array([0,0])
	out = np.zeros([rows,cols,3],dtype= np.uint8)
	for i in range(rows):
		for j in range(cols):
			#X = X.astype(np.uint8)
			Y = np.array([j,i]) - np.array(B)
			bool,X = cv2.solve(matrix,Y,X)
			nx = int(X[0,0])
			ny = int(X[1,0])
			if nx>0 and nx < cols and ny>0 and ny < rows:
				out[i,j] = input[ny,nx]
	#out = out.astype(np.uint8)
	cv2.imshow("salida",out)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def puntos(i,j,X):
	Shear = np.dot(X,[[j],[i]])
	nx = int(Shear[0,0])
	ny = int(Shear[1,0])
	return ny,nx


def shear_rotacion1(input,rows,cols,angle):
	B = np.array([(1-ma.cos(angle))*(cols//2) - ma.sin(angle)*(rows//2),ma.sin(angle)*(cols//2) + (1-ma.cos(angle))*(rows//2)])
	S1 = np.array([[1 ,-ma.tan(angle/2)],[0,1]])
	S2 = np.array([[1 , 0 ],[ma.sin(angle),1]])
	X = np.dot(S1,S2)
	X = np.dot(X,S1)
	y2,x2 = puntos(0,cols,X)
	y3,x3 = puntos(rows,cols,X)
	y4,x4 = puntos(rows,0,X)
	new_w = x2+abs(x4)
	new_h = y3
	out = np.zeros([int(new_h),int(new_w),3],dtype= np.uint8)
	print([y2,x2])
	print([y3,x3])
	print([y4,x4])
	#correccion_x = abs(cols-correccion_x)
	for i in range(new_h):
		for j in range(new_w):
			Shear = np.dot(X,[[j],[i-y2]])
			nx = int(Shear[0,0]) 
			ny = int(Shear[1,0] ) 
			if nx>0 and nx< cols and ny>0 and ny<rows:
				out[i,j] = input[ny,nx]
	#out = out.astype(np.uint8)
	cv2.imshow("salida_shear",out)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def shear_rotacion01(input,rows,cols,angle):
	B = np.array([(1-ma.cos(angle))*(cols//2) - ma.sin(angle)*(rows//2),ma.sin(angle)*(cols//2) + (1-ma.cos(angle))*(rows//2)])
	S1 = np.array([[1 ,-ma.tan(angle/2)],[0,1]])
	S2 = np.array([[1 , 0 ],[ma.sin(angle),1]])
	X = np.dot(S1,S2)
	X = np.dot(X,S1)
	y2,x2 = puntos(0,cols,X)
	y3,x3 = puntos(rows,cols,X)
	y4,x4 = puntos(rows,0,X)
	new_w = x2+abs(x4)
	new_h = y3
	out = np.zeros([int(new_h),int(new_w),3],dtype= np.uint8)
	for i in range(rows):
		for j in range(cols):
			Shear = np.dot(X,[[j],[i]])
			nx = Decimal(Shear[0,0]).quantize(0,ROUND_HALF_UP)
			ny = Decimal(Shear[1,0]).quantize(0,ROUND_HALF_UP) 
			#if nx>0 and nx< cols and ny>0 and ny<rows:
			out[int(ny-1),int(nx)+abs(x4)-1] = input[i,j]
	#out = out.astype(np.uint8)
	cv2.imshow("salida_shear",out)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	cv2.destroyAllWindows()

def shear_advance(input,rows,cols,angle):
	B = np.array([(1-ma.cos(angle))*(cols//2) - ma.sin(angle)*(rows//2),ma.sin(angle)*(cols//2) + (1-ma.cos(angle))*(rows//2)])
	S1 = np.array([[1 ,-ma.tan(angle/2)],[0,1]])
	S2 = np.array([[1 , 0 ],[ma.sin(angle),1]])
	X = np.dot(S1,S2)
	X = np.dot(X,S1)
	y2,x2 = puntos(0,cols,X)
	y3,x3 = puntos(rows,cols,X)
	y4,x4 = puntos(rows,0,X)
	new_w = x2+abs(x4)
	new_h = y3
	alpha = -ma.tan(angle/2)
	beta = ma.sin(angle)
	out = np.zeros([int(new_h),int(new_w),3],dtype= np.uint8)
	for i in range(rows):
		shear = alpha*i
		shear = ma.floor(shear)
		for j in range(cols):
			nx = j+shear
			ny = i
			ny = ma.floor(nx*beta)+ny
			nx = nx + ma.floor(ny*alpha)
			#print(nx)
			#ny = Decimal(ny).quantize(0,ROUND_HALF_UP)
			#if nx>0 and nx< cols and ny>0 and ny<rows:
			out[int(ny),int(nx)+abs(x4)-1] = input[i,j]
	#out = out.astype(np.uint8)
	cv2.imshow("salida_shear",out)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def make_matrix_angle(angle):
	Bx = (1-ma.cos(angle))*(cols//2) - ma.sin(angle)*(rows//2)
	By = ma.sin(angle)*(cols//2) + (1-ma.cos(angle))*(rows//2)
	M = np.float32([[ma.cos(angle),ma.sin(angle),Bx],[-ma.sin(angle),ma.cos(angle),By]])
	return M
def make_matrix_angle_AB(angle):
	#Bx = (1-ma.cos(angle))*(cols//2) - ma.sin(angle)*(rows//2)
	#By = ma.sin(angle)*(cols//2) + (1-ma.cos(angle))*(rows//2)
	M = np.array([[ma.cos(angle),ma.sin(angle)],[-ma.sin(angle),ma.cos(angle)]])
	B = np.array([(1-ma.cos(angle))*(cols//2) - ma.sin(angle)*(rows//2),ma.sin(angle)*(cols//2) + (1-ma.cos(angle))*(rows//2)])
	return M,B

img = cv2.imread("Julio.jpg")
rows, cols , channels = img.shape
Angle = ma.pi*(30/180)
#M = np.int32([[1,0,100],[0,1,20]])
##M = np.float32([[1,0.2,0],[0.1,1,0]])
#my_warpAffine(img,M,rows,cols)
M,b = make_matrix_angle_AB(Angle)



#src_points = np.float32([[1,0], [cols-1,0], [0,rows-1]])
#dst_points = np.float32([[0,1], [cols*2,0], [0,rows*2]])
#M = cv2.getAffineTransform(src_points, dst_points)
shear_rotacion1(img,rows,cols,Angle)


