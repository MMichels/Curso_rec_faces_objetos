'''
    utilizando o algoritmo HOG (que cria um vetor de diferencas nas imagens),
    podemos obter resultados com melhor precisao e menos falsos positivos;
    o HOG, percorre a imagem pixel e pixel, classificando a diferenteça de cor e magnitude da diferença
    com os pixels ao redor, isso cria uma imagem 'vetorizada' da imagem original

'''
import cv2
import dlib

imagem = cv2.imread("../Imagens-e-recursos/fotos/grupo.0.jpg")
# Cria o objeto que arnazeba o metodo de detecção de faces pre-compilada do Dlib
detector = dlib.get_frontal_face_detector()
subdetector = ['Olhar a frente', 'Vista esquerda', 'Vista direita',
               'A frente girando a esquerda', 'A frente virando a direita']
# metodo de deteccao__comparacao aprimorado, com pontuacao e a subdeteccao utilizada, alem do bounding box.
# 0.2 = nivel de confiabilidade
facesDetectadas, pontuacao, idx = detector.run(imagem, 2, 0.2)
print('Faces detectadas:', len(facesDetectadas))
for index, face in enumerate(facesDetectadas):
    texto = str('Confiabilidade face ' + str(index) + ': ' +str(pontuacao[index])
                + 'Moto deteccao__comparacao: ' + subdetector[int(idx[index])])
    print(texto)
    e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
    cv2.rectangle(imagem, (e, t), (d, b), (0, 0, 255), 1)
    cv2.putText(imagem, texto, (e, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), lineType=cv2.LINE_AA)

cv2.imshow('Resultado hog', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()
