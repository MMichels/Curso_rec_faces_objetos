import cv2
import dlib

imagem = cv2.imread('../Imagens-e-recursos/fotos/grupo.0.jpg')
detector = dlib.cnn_face_detection_model_v1('../Imagens-e-recursos/recursos/mmod_human_face_detector.dat')
facesDectadas = detector(imagem, 1)
tela = dlib.image_window()
retangulos = dlib.rectangles()

print("Number of faces detected: {}".format(len(facesDectadas)))
for i, d in enumerate(facesDectadas):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
        i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))

retangulos.extend([d.rect for d in facesDectadas])

tela.clear_overlay()
tela.set_image(imagem)
tela.add_overlay(retangulos)
dlib.hit_enter_to_continue()