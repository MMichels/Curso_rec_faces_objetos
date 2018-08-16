'''
    utilizando o algoritmo HOG (que cria um vetor de diferencas nas imagens),
    podemos obter resultados com melhor precisao e menos falsos positivos;
    o HOG, percorre a imagem pixel e pixel, classificando a diferenteça de cor e magnitude da diferença
    com os pixels ao redor, isso cria uma imagem 'vetorizada' da imagem original

'''
import cv2
import dlib

imagem = cv2.imread("../../Imagens-e-recursos/fotos/grupo.5.jpg")
# Cria o objeto que arnazeba o metodo de detecção de faces pre-compilada do Dlib
detector = dlib.get_frontal_face_detector()
# retorna um vator com valores ((left, top), (right, butto)), diferente do openCv.
# parametro 2 indica uma multiplicação do tamanho da imagem (cuidado, ocupa mais memoria)
# util para imagens com faces pequenas
facesDetectadas = detector(imagem, 2)
print('Faces detectadas: ', len(facesDetectadas))

# Como utilizaremos o metodo do OpenCv para ccriar os retangulos nas imagens, é necessario realizar uma conversao.
for face in facesDetectadas:
    esquerda = int(face.left())
    direita = int(face.right())
    topo = int(face.top())
    inferior = int(face.bottom())
    cv2.rectangle(imagem, (esquerda, topo), (direita, inferior), (0, 255, 0), 2)

cv2.imshow('Resultado hog', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()