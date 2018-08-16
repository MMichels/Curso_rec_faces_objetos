'''

    Neste exemplo sera treinado um algoritmo para reconhecer o ronald e a nancy em fotos
    serao treinados com os arquivos que estao na pasta '../../Imagens-e-recursos/fotos/treinamento/'
    e testados com as demais fotos de exemplo.

'''

import os
import glob
import _pickle as cPickle
import numpy as np
import dlib
import cv2

# cria o objeto de deteccao de faces
detectorFace = dlib.get_frontal_face_detector()
# abre o identificador de pontos com um arquivo treinado
detectorPontos = dlib.shape_predictor("../../Imagens-e-recursos/recursos/shape_predictor_68_face_landmarks.dat")
reconhecimentoFacial = dlib.face_recognition_model_v1('../../Imagens-e-recursos/recursos/dlib_face_recognition_resnet_model_v1.dat')

indice = {}
idx = 0
descritoresFaciais = None

# Extrai o descritor facial de cada uma das imagens salvas na pasta passada como parametro
for arquivo in glob.glob(os.path.join('../../Imagens-e-recursos/fotos/treinamento', '*.jpg')):
    if arquivo.count('mateus') == 0:
        img = cv2.imread(arquivo)
        facesDetectadas = detectorFace(img, 1)
        numeroFacesDetectadas = len(facesDetectadas)
        # print(numeroFacesDetectadas)
        if numeroFacesDetectadas > 1:
            print('Há mais de uma face na imagem {}'.format(arquivo))
            exit(1)
        elif numeroFacesDetectadas < 1:
            print('Nenhuma face detectada em : {}'.format(arquivo))
            exit(1)

        for face in facesDetectadas:
            pontosFaciais = detectorPontos(img, face)
            # seleciona as caracteristicas mais importantes da imagem, seguindo o principio de CNN
            # basicamente ele separa os dados para a camada de entrada da CNN
            descritorFacial = reconhecimentoFacial.compute_face_descriptor(img, pontosFaciais)
            # print(format(arquivo))
            # print(len(descritorFacial))
            # print(descritorFacial)
            listaDescritorFacial = [df for df in descritorFacial]
            npArrayDescritorFacial = np.asarray(listaDescritorFacial, dtype=np.float64)
            # aumenta o vetor em 1 dimencao
            npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]
            if descritoresFaciais is None:
                descritoresFaciais = npArrayDescritorFacial
            else:
                descritoresFaciais = np.concatenate((descritoresFaciais, npArrayDescritorFacial), axis=0)

            indice[idx] = arquivo
            idx += 1

#print('tamanho: {} Formato: {}'.format(len(descritoresFaciais), descritoresFaciais.shape))
np.save('../../Imagens-e-recursos/recursos/descritores_rn.npy', descritoresFaciais)
with open('../../Imagens-e-recursos/recursos/indices_rn.pickle', 'wb') as f:
    cPickle.dump(indice, f)

cv2.destroyAllWindows()
