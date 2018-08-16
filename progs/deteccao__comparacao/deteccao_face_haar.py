import cv2

# abre a imagem em escala de cinzas
imagem = cv2.imread("../Imagens-e-recursos/fotos/grupo.7.jpg")
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
classificador = cv2.CascadeClassifier('../Imagens-e-recursos/recursos/haarcascade_frontalface_default.xml')
# passa o detector na imagem e retorna os valores [x, y, width, height] de cada face detectada
# scaleFactor = escala minima de detecção, serve para aprimorar a detecção e é definido em teste e erro
facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.325)
for (x, y, largura, altura) in facesDetectadas:
    cv2.rectangle(imagem, (x, y), (x + largura, y + altura), (0, 255, 0), 1)


cv2.imshow('Detector Haar', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()