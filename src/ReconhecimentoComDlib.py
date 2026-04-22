
import dlib
import cv2
import os

import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

caminho_modelo_68 = os.path.join(BASE_DIR, "..", "models", "shape_predictor_68_face_landmarks.dat")
caminho_descritor = os.path.join(BASE_DIR, "..", "models", "dlib_face_recognition_resnet_model_v1.dat")
caminho_dataset = os.path.join(BASE_DIR, "..", "data", "yalefaces", "train")


detecctorFaceDlib = dlib.get_frontal_face_detector()
detecctorPontosFacial = dlib.shape_predictor(caminho_modelo_68)
descritorFacial = dlib.face_recognition_model_v1(caminho_descritor)



index = {}
idx = 0
descritoresFacial = None

paths = [os.path.join(caminho_dataset,f) for f in os.listdir(caminho_dataset)]
for path in paths :
    imagem = Image.open(path).convert('RGB')
    imagem_np = np.array(imagem,'uint8')
    deteccoesDlib = detecctorFaceDlib(imagem_np,2)

    for face in deteccoesDlib :

        l,t,r,b = face.left(),face.top(),face.right(),face.bottom()
        cv2.rectangle(imagem_np,(l,t),(r,b),(0,0,255),2)

        pontos = detecctorPontosFacial(imagem_np, face)
        for ponto in pontos.parts():
            cv2.circle(imagem_np, (ponto.x, ponto.y), 2, (0, 255, 0), 1)

        descritor = descritorFacial.compute_face_descriptor(imagem_np, pontos)
        descritor = [f for f in descritor]
        descritor = np.asarray(descritor,dtype =np.float64)
        descritor = descritor[np.newaxis, :]

        if descritoresFacial is None :
            descritoresFacial = descritor
        else :
            descritoresFacial = np.concatenate((descritoresFacial,descritor),axis=0)

        index[idx] = path
        idx += 1

caminhoYalefacesTeste = os.path.join(BASE_DIR,"..","yalefaces","test")
confiancaMinima = 0.5
pathsTeste = [os.path.join(caminhoYalefacesTeste,f) for f in os.listdir(caminhoYalefacesTeste)]

for path in pathsTeste :
    imagem = Image.open(path).convert('RGB')
    imagem_np = np.array(imagem,'uint8')
    deteccoes = detecctorFaceDlib(imagem_np,2)

    for face in deteccoes :
        pontos = detecctorPontosFacial(imagem_np,face)
        descritor = descritorFacial.compute_face_descriptor(imagem_np,pontos)
        descritor = [f for f in descritor]
        descritor = np.asarray(descritor, dtype=np.float64)
        descritor = descritor[np.newaxis, : ]

        distancia = np.linalg.norm(descritor - descritoresFacial, axis = 1)
        indiceMinimo = np.argmin(distancia)
        distanciaMinima = distancia[indiceMinimo]

        if distanciaMinima <= confiancaMinima :
            nomePrevisao = int(os.path.split(index[indiceMinimo])[1].split('.')[0].replace('subject',''))
        else :
            print('Face não identificada !')

        nomeReal = int(os.path.split(path)[1].split('.')[0].replace('subject',''))

        cv2.putText(imagem_np,'Pred : ' + str(nomePrevisao), (10,30), cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0))
        cv2.putText(imagem_np,'Exp : '  + str(nomeReal), (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))

        cv2.imshow("Imagem", imagem_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


