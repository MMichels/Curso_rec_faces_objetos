import os, glob, dlib, cv2
import _pickle as cPickle
import numpy as np

# cria o objeto de deteccao de faces
detectorFace = dlib.get_frontal_face_detector()
# abre o identificador de pontos com um arquivo treinado
detectorPontos = dlib.shape_predictor("../../Imagens-e-recursos/recursos/shape_predictor_68_face_landmarks.dat")
reconhecimentoFacial = dlib.face_recognition_model_v1('../../Imagens-e-recursos/recursos/dlib_face_recognition_resnet_model_v1.dat')
indices = np.load('../../Imagens-e-recursos/recursos/indices_rn.pickle')
descritoresFaciais = np.load('../../Imagens-e-recursos/recursos/descritores_rn.npy')

for arquivo in glob.glob(os.path.join('../../Imagens-e-recursos/fotos', '*.jpg')):
    if arquivo.count('mateus') == 0:
        img = cv2.imread(arquivo)
        facesDetectadas = detectorFace(img, 2)
        for face in facesDetectadas:
            e, t, d, b = (int(face.left()), int(face.top()),
                          int(face.right()), int(face.bottom()))
            cv2.rectangle(img, (e, t), (d, b), (0, 255, 255), 2)
            pontosFaciais = detectorPontos(img, face)
            descritorFacial = reconhecimentoFacial.compute_face_descriptor(img, pontosFaciais)
            # para reconhecer a pessoa na imagem, é preciso repetir a mesma estrutura que foi usada no treinamento
            listaDescritorFacial = [fd for fd in descritorFacial]
            npArrayDescritorFacial = np.asarray((listaDescritorFacial), dtype=np.float64)
            npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]


        cv2.imshow("Detector HoG", img)
        cv2.waitKey(0)

cv2.destroyAllWindows()