import copy
import math
from collections import Counter
import numpy as np
import numpy.linalg as nplg
def RightOrder(values,matrix):
    sort_index=np.flip(np.argsort(values))

    matrixResult=copy.deepcopy(matrix)
    return matrixResult[:,sort_index]
def Scalar(vector1,vector2):
    suma=0
    for i in range(len(vector1)):
        suma+=vector1[i]*vector2[i]
    return suma


def Ortogonalizacja(Matrix):
    toOrthogonal=copy.deepcopy(Matrix)
    toOrthogonal[0]=np.array(toOrthogonal[0])
    vectortoSubstract = []
    for i in range(1,len(toOrthogonal)):
        vector=np.array(toOrthogonal[i])
        VectorsToSub=[0]*len(toOrthogonal[i])
        for k in range(i):

            scalar=Scalar(vector, toOrthogonal[k]) / Scalar(toOrthogonal[k], toOrthogonal[k])

            for s in range(len(toOrthogonal[k])):
                vectortoSubstract.append(toOrthogonal[k][s]*scalar)
            VectorsToSub=np.add(VectorsToSub,vectortoSubstract)
            vectortoSubstract=[]
        toOrthogonal[i]=np.subtract(toOrthogonal[i],VectorsToSub)

    return toOrthogonal;


def DoubleVEctors(EigenValues,EigenVectors):
    previous=float("inf")
    toOrthogonal=[]
    transposedVectors=np.transpose(copy.deepcopy(EigenVectors))
    for k in range(len(EigenValues)):
        if previous!=EigenValues[k]:
            Occur=np.count_nonzero(EigenValues==EigenValues[k])
            if Occur!=1:
                for s in range(Occur):
                    toOrthogonal.append(transposedVectors[s])

                Result=Ortogonalizacja(toOrthogonal)
                #print(Result)
            else:
                Result.append(transposedVectors[k])
        previous=EigenValues[k]
    return np.transpose(Result)

matrix=[[1,1,1,1],[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]]

EigenValues=np.array(nplg.eig(matrix)[0])
EigenVectors=np.array(nplg.eig(matrix)[1])
#print(EigenValues) #bez zmian

#print(EigenVectors) #be zzmian
#print("pwo")

EigenVectors=RightOrder(EigenValues,EigenVectors) #ustawianie wektorow w kolejnosc
EigenValues=np.flip(np.sort(EigenValues)) #sortowanie Eigenvalues
print(EigenValues)
print(EigenVectors)


result=DoubleVEctors(EigenValues,EigenVectors)
print(Scalar(result[1],result[2]))
#print(result)
#matrix=[[1,1,1,1],[0,0,0,1],[1,1,0,0]]
#print(Ortogonalizacja(matrix))
print(nplg.eig([[1,1],[0,1]]))
