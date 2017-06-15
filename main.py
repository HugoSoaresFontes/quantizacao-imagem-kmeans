# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy as sp
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from time import time

print "Início do processo"
t_inicio = time()
# Numero de clusters/cores
n_clusters = 32
# Carregamento da imagem
imagem = sp.misc.imread("gold_coast_australia.jpg")
# Converte de 0-255 para 0-1
imagem = np.array(imagem, dtype=np.float64) / 255
w, h, d = original_shape = tuple(imagem.shape)
# Verifica se os pixels tem dimensao 3, ou seja, rgb
assert d == 3

# Transforma a imagem em um nump array de duas dimensoes
imagem_array = np.reshape(imagem, (w * h, d))

print "Treinando o kmeans com um resumo do modelo"
t0 = time()
# Faz um resumo randomico dos dados, para agilizar o encontro dos clusters
imagem_array_resumo = shuffle(imagem_array, random_state=0)[:2000]
# Kmeans dos dados resumidos. Pode ser feito direto do image_array, o que aumenta o tempo, mas também aumenta precisao
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(imagem_array_resumo)
print "Processo feito em %0.3fs." % (time() - t0)

# Etiqueta de cada pixel da imagem
print "Predicao das cores com k-means"
t0 = time()
etiquetas = kmeans.predict(imagem_array)
print "Processo feito em %0.3fs." % (time() - t0)


def reconstruir_imagem(codebook, etiquetas, w, h):
    ''' Reconstrução da imagem com base no codebook (os clusters), nas etiquetas
        e na dimensão da imagem original
    '''
    d = codebook.shape[1]
    imagem = np.zeros((w, h, d))
    indice = 0
    for i in range(w):
        for j in range(h):
            imagem[i][j] = codebook[etiquetas[indice]]
            indice += 1
    return imagem

def vetor_diferenca(original, reconstruida):
    ''' Retorna um vetor com a diferença de cada pixel da imagem original
        em relacação a imagem reconstru
    '''
    w, h, d = imagem.shape
    vetor_diferenca = np.zeros((w*h*d))
    index = 0
    for i in range(w-10):
        for j in range(h):
            for k in range(d):
                vetor_diferenca[index] += abs(original[i,j,k] - reconstruida[i,j,k])
                index += 1
    return vetor_diferenca

imagem_reconstruida = reconstruir_imagem(kmeans.cluster_centers_, etiquetas, w, h)
print "Fim do processo. Tempo total: %0.3fs." % (time() - t_inicio)

plt.figure("Original")
plt.clf()
plt.title('Imagem Original')
plt.imshow(imagem)

plt.figure("Reconstruida")
plt.clf()
plt.title('Quantizada e reconstruida com ' + str(n_clusters) + ' Clusters')
plt.imshow(imagem_reconstruida)

# Plot das Clusters/Cores
fig = plt.figure("Cores dos clusters")
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Cores dos Clusters")
ax.set_xlabel('R - Vermelho')
ax.set_ylabel('G - Verde')
ax.set_zlabel('B - Azul')
for cor in kmeans.cluster_centers_:
    c = [0, 0, 0]
    for i in range(3):
        c[i] = cor[i]*255
    c = tuple(c)
    # Convertendo para hexadecimal
    c = '#%02x%02x%02x' % c
    ax.scatter(cor[0]*255, cor[1]*255, cor[2]*255, c=c, s=160)

plt.show()
