'''
	Realiza a detectao de 68 pontos na face 'dentro do bouding box'
	tentam localizar: queixo, labios, sobrancelhas, olhos e nariz

'''
import dlib
from cv2 import FONT_HERSHEY_COMPLEX_SMALL, putText, polylines, circle, imread, imshow, waitKey, destroyAllWindows
from numpy import array, int32

_VERDE = (0, 255, 0)
_FONTE = FONT_HERSHEY_COMPLEX_SMALL

''' Recebe uma imagem e a sua respectiva lista de pontos e printa os pontos na imagem utilizando a funcao
    cv2.circle
    :param img imagem aberta com a funcao cv2.imread() ou semelhante dlib.load_rgb_image()
    :param pontosFaciais lista de pontos obtida com a funcao dlib.shape_predictor()   
    :param desenhaNumeros define se serao desenhados circulos ou pontos, padrao = False
'''
def imprimePontos(img, pontosFaciais, desenhaNumeros=False, desenhaLinhas=False):
    if desenhaNumeros:
        # desenha os numeros
        for i, p in enumerate(pontosFaciais.parts()):
            # (img, string_numeros, ponto_central, fonte, tamanho_fonte, cor_fronte, borda
            putText(img, str(i), (p.x, p.y), _FONTE, .55, _VERDE, 1)
        return
    if desenhaLinhas:
        # [pontos[ponto_inicial, ponto_final, linha_fechada]]
        pontosInteresse = [[0, 16, False],   # linha do queixo
                          [17, 21, False],  # sobrancelha_direita
                          [22, 26, False],  # sobrancelha_esquerda
                          [27, 30, False],  # ponte_nasal
                          [31, 35, True],  # nariz_inferior
                          [36, 41, True],   # olho_esquerdo
                          [42, 46, True],   # olho_direito
                          [48, 59, True],   # labio externo
                          [60, 67, True]]   # labio_interno
        for k in range(0, len(pontosInteresse)):
            pontos = []
            for i in range(pontosInteresse[k][0], pontosInteresse[k][1] + 1):
                ponto = [pontosFaciais.part(i).x, pontosFaciais.part(i).y]
                pontos.append(ponto)
            pontos = array(pontos, dtype=int32)
            polylines(img, [pontos], pontosInteresse[k][2], _VERDE, 2)
        return

    for p in pontosFaciais.parts():
        # (imagem, ponto cental, raio em pixels, cor, tipo_linha
        circle(img, (p.x, p.y), 2, _VERDE, 2)


img = imread('../../Imagens-e-recursos/fotos/treinamento/mateus.0.1.jpg', 1)
# cria o objeto de deteccao de faces
detectorFace = dlib.get_frontal_face_detector()
# abre o identificador de pontos com um arquivo treinado
detectorPontos = dlib.shape_predictor("../../Imagens-e-recursos/recursos/shape_predictor_68_face_landmarks.dat")
# detecta as faces na imagem
facesDetectadas = detectorFace(img, 2)
for face in facesDetectadas:
    # detecta os pontos dentro de cada face detectada
    ponts = detectorPontos(img, face)
    # print da quantidade de pontos:
    print(len(ponts.parts()))
    # print dos pontos detectados
    print(ponts.parts())
    # desenha os pontos na imagem
    imprimePontos(img, ponts, False, True)

imshow('Ronald', img)
waitKey(0)
destroyAllWindows()
