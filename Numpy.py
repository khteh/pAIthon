import numpy
from pathlib import Path

def oneDArray(n: int):
    one = numpy.array(range(n))
    print(f"{one} shape: {one.shape}\n")
def twoDArray(n: int):
    cols = (int)(n / 2)
    two = numpy.array([range(cols), range(cols, 2*cols)])
    print(f"{two} shape: {two.shape}\n")
def threeDArray(n: int):
    cols = (int)(n / 3)
    three = numpy.array([range(cols), range(cols, 2*cols), range(2*cols, 3*cols)])
    print(f"{three} shape: {three.shape}\n")
def multiDArray(i:int, j: int, k:int):
    multi = numpy.zeros((i,j,k))
    print(f"{multi} shape: {multi.shape}\n")
def csvArray(i:int, j: int, k: int, path):
    print(f"=== csvArray.__name__ ===")
    data = numpy.zeros((i,j,k))
    print(f"id: {id(data)}")
    for index, file in enumerate(Path.cwd().glob(path)):
        print(f"index: {index}, file: {file.name}")
        data[index] =  numpy.loadtxt(file, delimiter=",")
    print(f"data ({id(data)}): {data}\n")
def csvArrayInsert(i:int, j: int, k: int, path, insertIndex: int, insertValue: int, insertAxis:int, insertFile):
    print(f"=== csvArrayInsert.__name__ ===")
    data = numpy.zeros((i,j,k))
    print(f"id: {id(data)}")
    for index, file in enumerate(Path.cwd().glob(path)):
        print(f"index: {index}, file: {file.name}")
        data[index] =  numpy.loadtxt(file, delimiter=",")
    print(f"data ({id(data)}): {data}")
    data = numpy.insert(arr=data, obj=insertIndex, values=insertValue, axis=insertAxis)
    print(f"After inserting a row: data ({id(data)}): {data}")
    data[i-1] = numpy.loadtxt(insertFile, delimiter=",")
    print(f"Load from new csv: data ({id(data)}): {data}\n")
oneDArray(10)
twoDArray(20)
threeDArray(30)
multiDArray(3,4,5)
csvArray(3,2,3, "data/file?.csv")
"""
https://stackoverflow.com/questions/46855793/understanding-axes-in-numpy
In numpy, axis ordering follows zyx convention, instead of the usual (and maybe more intuitive) xyz.
"""
csvArrayInsert(4,2,3,"data/file?.csv",2,0,1,"data/file4_extra_row.csv")