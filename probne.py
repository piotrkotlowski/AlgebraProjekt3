import cmath
from matplotlib.image import imread
import copy
import math
import numpy as np
import numpy.linalg as nplg


def Scalar(vector1, vector2):
    suma = 0
    for i in range(len(vector1)):
        suma += vector1[i] * vector2[i]
    return suma


def Ortogonalizacja(Matrix):
    toOrthogonal = copy.deepcopy(Matrix)
    toOrthogonal[0] = np.array(toOrthogonal[0])
    vectortoSubstract = []
    for i in range(1, len(toOrthogonal)):
        vector = np.array(toOrthogonal[i])
        VectorsToSub = [0] * len(toOrthogonal[i])
        for k in range(i):

            scalar = Scalar(vector, toOrthogonal[k]) / Scalar(toOrthogonal[k], toOrthogonal[k])

            for s in range(len(toOrthogonal[k])):
                vectortoSubstract.append(toOrthogonal[k][s] * scalar)
            VectorsToSub = np.add(VectorsToSub, vectortoSubstract)
            vectortoSubstract = []
        toOrthogonal[i] = np.subtract(toOrthogonal[i], VectorsToSub)

    return toOrthogonal;


def DoubleVEctors(EigenValues, EigenVectors):
    previous = float("inf")
    toOrthogonal = []
    Result = []
    transposedVectors = np.transpose(copy.deepcopy(EigenVectors))
    for k in range(len(EigenValues)):
        if previous != EigenValues[k]:
            Occur = np.count_nonzero(EigenValues == EigenValues[k])
            if Occur != 1:
                for s in range(Occur):
                    toOrthogonal.append(transposedVectors[s])
                Result = Ortogonalizacja(toOrthogonal)
            else:
                Result.append(transposedVectors[k])
        previous = EigenValues[k]
    return np.transpose(Result)


def GiveMatrix(input):
    matrix = np.matmul(np.transpose(input), input)
    return matrix


def ScientificToFloatVector(vector):
    for i in range(len(vector)):
        # vector[i]=float(vector[i]) #Problem tutaj z konwertowaniem
        vector[i] = '{:f}'.format(vector[i])
    return vector


def EigenToSingular(matrix):
    matrixResult = [0] * len(matrix)
    for i in range(len(matrix)):
        matrixResult[i] = cmath.sqrt(matrix[i])
    return matrixResult


def Norm(vector):
    ResultNorm = 0
    for k in range(len(vector)):
        ResultNorm += vector[k] * vector[k]
    return cmath.sqrt(ResultNorm)


def NormForEigenVectors(matrix):
    # print(matrix)
    matrixOperation = np.transpose(copy.deepcopy(matrix))
    matrixResult = copy.deepcopy(matrixOperation)
    # print(matrixOperation)
    for i in range(len(matrixOperation)):
        norm = Norm(matrixOperation[i])
        for j in range(len(matrixOperation[i])):
            matrixResult[i][j] = matrixOperation[i][j] / norm
    matrixResult = np.transpose(matrixResult)
    return matrixResult


def matrixU(singularvalues, A, eigenvectors):
    matrixResult = []
    RightFormatEvectors = np.transpose(copy.deepcopy(eigenvectors))
    for k in range(len(A)):
        matrixResult.append(np.array(np.matmul(A, RightFormatEvectors[k])) / singularvalues[k])
    return np.transpose(matrixResult)


def SingValuestoMatrix(input, singularvalues):
    singularvaluesSorted = copy.deepcopy(singularvalues)
    singularvaluesSorted = np.flip(np.sort(singularvaluesSorted))
    Result = np.zeros((len(input), len(input[0])))
    for k in range(len(input)):
        Result[k][k] = singularvaluesSorted[k]
    return Result


def RightOrder(values, matrix):
    sort_index = np.flip(np.argsort(values))

    matrixResult = copy.deepcopy(matrix)
    return matrixResult[:, sort_index]


def ReverseMatrix(Matrixinput):
    Matrix = copy.deepcopy(Matrixinput)
    for i in range(len(Matrix)):
        Matrix[i][i] = 1 / Matrix[i][i]
    return np.transpose(Matrix)


def Noises(List, k):
    for i in range(k, len(List), 1):
        List[i] = 0

    return List


def main():
    a = imread("skimage_astronaut.png")

    matrix = GiveMatrix(a)
    EigenValues = np.array(nplg.eig(matrix)[0])
    EigenVectors = np.array(nplg.eig(matrix)[1])

    # Liczenie singular values
    # EigenValues=ScientificToFloatVector(EigenValues)

    SingularValues = EigenToSingular(EigenValues)

    # SingularValues=np.flip(np.sort(SingularValues))
    #MatrixSingValus = SingValuestoMatrix(a, SingularValues)

    # Liczenie v transponowego

    EigenVectors = RightOrder(SingularValues, EigenVectors)
    EigenValues = np.flip(np.sort(EigenValues))
    EigenVectors = DoubleVEctors(EigenValues, EigenVectors)
    EigenVectors = NormForEigenVectors(EigenVectors)

    # Liczenie U
    SingularValuesforU = np.flip(np.sort(SingularValues))
    U = (matrixU(SingularValuesforU, a, EigenVectors))

    # print(U)
    # print(MatrixSingValus)
    # print(np.transpose(EigenVectors))

    # Noises
    k=int(input("Ile chcesz zostawić wartosci osobliwych?"))
    MatrixSingValusNoise = SingValuestoMatrix(a, Noises(SingularValues, k)) #Zmienna k

    firstMultiNoise = np.matmul(U, MatrixSingValusNoise)
    resultNoise = np.matmul(firstMultiNoise, np.transpose(EigenVectors))

   # firstMulti = np.matmul(U, MatrixSingValus)
   # result = np.matmul(firstMulti, np.transpose(EigenVectors))

    from matplotlib import pyplot as plt

    plt.imshow(resultNoise.real, interpolation='nearest',cmap="gray")
    #plt.imshow(result.real,interpolation='nearest',cmap="gray")
    plt.show()


if __name__ == '__main__':
    main()
