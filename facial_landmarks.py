# USAGE
# python facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --image images/example_01.jpg 

# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

# constrói o analisador de argumentos e analisa os argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())


# inicialize o detector de rosto do dlib (baseado em HOG) e crie
# o indicador de marco facial

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# carrega a imagem de entrada, redimensione-a e converta-a em escala de cinza
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# detectar rostos na imagem em escala de cinza
rects = detector(gray, 1)


# laço nas detecções de rosto
for (i, rect) in enumerate(rects):
# determine os pontos de referência faciais da região do rosto e, em seguida,
# converte o marco facial (x, y) -coordena em um NumPy
# array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)


# converter o retângulo do dlib em uma caixa delimitadora no estilo OpenCV
# [ou seja, (x, y, w, h)] e desenhe a caixa delimitadora da face
	(x, y, w, h) = face_utils.rect_to_bb(rect)
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


# mostra o número do rosto
	cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# loop sobre as coordenadas (x, y) para os pontos de referência faciais
# e desenhe-os na imagem
	for (x, y) in shape:
		cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

# mostra a imagem de saída com as detecções de rosto + pontos faciais
cv2.imshow("Output", image)
cv2.waitKey(0)