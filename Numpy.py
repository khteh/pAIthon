import numpy, math, tensorflow as tf
from pathlib import Path
from utils.GPU import InitializeGPU
import numpy.lib.recfunctions as reconcile
from numpy.random import Generator, PCG64DXSM
import matplotlib.pyplot as plt
rng = Generator(PCG64DXSM())
"""
https://numpy.org/doc/stable/user/basics.indexing.html
Numpy main data type is ndarray
shape (n,m,...) goes from outer-most (n) to inner-most dimension (m,...).
numpy.random.seed() is used to create reproducible random numbers aligned with the seed.
numpy.random.randint(a, size=(n,m)): Generates random numbers of range[0, a) and shape=(n,m)
numpy.random.random((n,m)): Generates random numbers of range[0,1) of shape=(n,m)
// : Floor division - Removes the decimals / round down
numpy.var(): Variance = A measure of the average degree to which each number is diff to the mean. Higher variance: wider range of numbers
numpy.std(): Standard deviation = A measure of how spread out a group of numbers is from the mean. = sqrt(numpy.var())
.T: Transpose reverses the axes. Ex: (2,3,4) -> (4,3,2)
numpy.argsort(): Return the indices which will sort the ndarray
"""
def oneDArray(n: int):
    print(f"=== {oneDArray.__name__} ===")
    one = numpy.array(range(n))
    print(f"{one} shape: {one.shape}\n")
    assert one.shape[0] == n
    pi = numpy.full(len(one), math.pi)
    print(f"{pi} shape: {pi.shape}\n")
    assert pi.shape[0] == n
    assert pi.shape == one.shape
    assert numpy.all(pi == math.pi)
    one = numpy.arange(n) # numpy.arange() creates an array of consecutive, equally-spaced values within a given range
    print(f"{one} shape: {one.shape}\n")
    print(f"index-0 of shape: {one.shape[0]}, index--1 of shape: {one.shape[-1]}")
    one = numpy.arange(n).reshape((-1, 1)) # One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.
    print(f"{one} shape: {one.shape}\n")

def twoDArray(n: int):
    print(f"=== {twoDArray.__name__} ===")
    cols = (int)(n / 3)
    two = numpy.array([range(cols), range(cols, 2*cols), range(2*cols, 3*cols)])
    print(f"{two} ndim: {two.ndim}, size: {two.size}, shape: {two.shape}")
    numbers = numpy.array([[1,3,5], [7,9,11]])
    print(f"ndim: {numbers.ndim}, size: {numbers.size}, shape: {numbers.shape}")
    # The axis=0 argument specifies that the accumulation should occur along the rows, effectively accumulating values down each column.
    aggregate0 = numpy.sum(numbers, axis=0) # Each column collapsed into a single row
    aggregate1 = numpy.sum(numbers, axis=1) # Each row collapsed into a single column
    print(f"Aggregates axis-0: {aggregate0}, axis-1: {aggregate1}")

def Matrix():
    print(f"\n=== {Matrix.__name__} ===")
    #vector indexing operations on matrices
    a = numpy.arange(6).reshape(-1, 2)   #reshape is a convenient way to create matrices
    print(f"a.shape: {a.shape}, \na= {a}")

    #access an element
    print(f"\na[2,0].shape:   {a[2, 0].shape}, a[2,0] = {a[2, 0]},     type(a[2,0]) = {type(a[2, 0])} Accessing an element returns a scalar\n")

    #access a row
    print(f"a[2].shape:   {a[2].shape}, a[2]   = {a[2]}, type(a[2])   = {type(a[2])}")

    #vector 2-D slicing operations
    a = numpy.arange(20).reshape(-1, 10)
    print(f"a = \n{a}")

    #access 5 consecutive elements (start:stop:step)
    print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

    #access 5 consecutive elements (start:stop:step) in two rows
    print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

    # access all elements
    print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)

    # access all elements in one row (very common usage)
    print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "a 1-D array")
    # same as
    print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")

def multiDArray(i:int, j: int, k:int):
    """
    For 3-D, axis-0 is z-axis, axis-1 is rows, axis-2 is columns
    """
    print(f"\n=== {multiDArray.__name__} ===")
    multi = numpy.zeros((i,j,k))
    print(f"{multi} shape: {multi.shape}\n")
    multi = numpy.arange(10)
    #print(f"[1,3]: {multi[1,3]}")
    multi.shape = (2,5)
    # Note that if one indexes a multidimensional array with fewer indices than dimensions, one gets a subdimensional array. For example:
    print(f"[0]: {multi[0]}, [1]: {multi[1]}")
    # [1,3] [axis-0 index-1, axis-1 index-3]
    print(f"[1,3] [axis-0 index-1, axis-1 index-3]: {multi[1,3]}")
    # z-axis (axis-0) consists of 2 elements; y-axis (axis-1) consists of 3 elements; x-axis (axis-2) consists of 1 element:
    multi = numpy.array([[[1],[2],[3]], [[4],[5],[6]]])
    print(f"ndim: {multi.ndim}, size: {multi.size}, shape: {multi.shape}")
    # If the number of objects in the selection tuple is less than N, then : is assumed for any subsequent dimensions. For example:
    print(f"axis-0 0:1 => {multi[:1]}")
    print(f"axis-0 1:2 => {multi[1:2]}")
    print(f"axis-0 0:2 => {multi[:2]}")
    print(f"axis-0 element-0, axis-1 0:1 => {multi[0, :1]}")
    print(f"axis-0 element-0, axis-1 1:2 => {multi[0, 1:2]}")
    print(f"axis-0 element-0, axis-1 0:2 => {multi[0, :2]}")
    print(f"axis-0 element-1, axis-1 0:1 => {multi[1, :1]}")
    print(f"axis-0 element-1, axis-1 1:2 => {multi[1, 1:2]}")
    print(f"axis-0 element-1, axis-1 0:2 => {multi[1, :2]}")

def AdvancedIndexing():
    """
    https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing
    It's using idx to index (select) from the first dimension (rows) of centroids. The ,: part makes it clear that it will return all values along the second dimension.
    """
    centroids = numpy.array([[0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                    [0.7,0.8,0.9]])
    idx = numpy.array([2,1,0,1,2,0,2,0,1])
    X_recovered = centroids[idx, :]
    assert X_recovered.all() == numpy.array([[0.7, 0.8, 0.9],
                                        [0.4, 0.5, 0.6],
                                        [0.1, 0.2, 0.3],
                                        [0.4, 0.5, 0.6],
                                        [0.7, 0.8, 0.9],
                                        [0.1, 0.2, 0.3],
                                        [0.7, 0.8, 0.9],
                                        [0.1, 0.2, 0.3],
                                        [0.4, 0.5, 0.6]]).all()
    print(f"centroid: {centroids}, shape: {centroids.shape}")
    print(f"idx: {idx}, shape: {idx.shape}")
    print(f"centroids[idx, :]: {X_recovered}, shape: {X_recovered.shape}")

def Broadcasting():
    """
    Z = XW + b utilized NumPy broadcasting to expand the vector b. If you are not familiar with NumPy Broadcasting, this short tutorial is provided.
    XW is a matrix-matrix operation with dimensions (𝑚,j1)(j1,j2) which results in a matrix with dimension (𝑚,j2). To that, we add a vector b with dimension (1,j2). b must be expanded to be a (𝑚,j2) matrix for this element-wise operation to make sense. This expansion is accomplished for you by NumPy broadcasting.

    Broadcasting applies to element-wise operations.
    Its basic operation is to 'stretch' a smaller dimension by replicating elements to match a larger dimension.

    More specifically: When operating on two arrays, NumPy compares their shapes element-wise. It starts with the trailing (i.e. rightmost) dimensions and works its way left. Two dimensions are compatible when

    they are equal, or
    one of them is 1
    If these conditions are not met, a ValueError: operands could not be broadcast together exception is thrown, indicating that the arrays have incompatible shapes. The size of the resulting array is the size that is not 1 along each axis of the inputs.    
    """
    a = numpy.array([1,2,3]).reshape(-1,1)  #(3,1)
    b = 5
    result = a + b
    assert result.shape == (3,1)
    assert (result == numpy.array([[6],[7],[8]])).all()
    result = a * b
    assert result.shape == (3,1)
    assert (result == numpy.array([[5],[10],[15]])).all()
    a = numpy.array([1,2,3,4]).reshape(-1,1) # (4,1)
    b = numpy.array([1,2,3]).reshape(1,-1) # (1,3)
    result = a + b
    assert result.shape == (4,3)
    assert (result == numpy.array([
                        [2, 3, 4],
                        [3, 4, 5],
                        [4, 5, 6],
                        [5, 6, 7]])).all()

def ConcatenateSliceObjects():
    """
    https://numpy.org/doc/stable/reference/generated/numpy.c_.html
    """
    x = numpy.arange(0, 20, 1)
    print(f"ndim: {x.ndim}, size: {x.size}, shape: {x.shape}")
    X = numpy.c_[x, x**2, x**3]   #<-- added engineered feature
    print(f"ndim: {X.ndim}, size: {X.size}, shape: {X.shape}")

def csvArray(i:int, j: int, k: int, path):
    print(f"\n=== {csvArray.__name__} ===")
    data = numpy.zeros((i,j,k))
    print(f"id: {id(data)}, ndim: {data.ndim}, size: {data.size}, shape: {data.shape}")
    """
    https://stackoverflow.com/questions/79429728/python-multidimension-array-axis-and-index-to-load-csv-file-into
    Replacing the i-th element in axis 1 would be data[:, i] =, in axis 2 would be data[:, :, i] =. Here :, is shorthand for "all entries in axis" (i.e. all entries in axis 0 for data[:, i], all entries in axis 0 and 1 for data[:, :, i]) 
    """
    for index, file in enumerate(Path.cwd().glob(path)):
        print(f"index: {index}, file: {file.name}")
        data[index] =  numpy.loadtxt(file, delimiter=",")
    print(f"data ({id(data)}): {data}")

def csvArrayInsert(i:int, j: int, k: int, path, insertIndex: int, insertValue: int, insertAxis:int, insertFile):
    print(f"\n=== {csvArrayInsert.__name__} ===")
    data = numpy.zeros((i,j,k))
    print(f"id: {id(data)}")
    for index, file in enumerate(Path.cwd().glob(path)):
        print(f"index: {index}, file: {file.name}")
        data[index] =  numpy.loadtxt(file, delimiter=",")
    print(f"data ({id(data)}), ndim: {data.ndim}, size: {data.size}, shape: {data.shape}")
    print(data)
    data = numpy.insert(arr=data, obj=insertIndex, values=insertValue, axis=insertAxis)
    print(f"After inserting a row: data ({id(data)}): {data}")
    data[i-1] = numpy.loadtxt(Path(insertFile), delimiter=",")
    print(f"Load from new csv: data ({id(data)}): {data}")

def structuredArray():
    print(f"\n=== {structuredArray.__name__} ===")
    books = numpy.array(
        [
            ("The Subtle Art of Not Giving a F**k", 1.2, 3),
            ("Never Split The Difference", 3.4, 1),
            ("48 Laws of Power", 5.6, 4),
            ("Mindset", 7.8, 2)
        ], dtype = [
            ("title", "U128"),
            ("price", "f4"),
            ("position", "i4")
        ]
    )
    print(f"Books: {books}")
    print(f"Books[title]: {books["title"]}")
    sortedbooks = numpy.sort(books, order="position")
    print(f"Books sorted on position: {sortedbooks}")
    print(f"Books sorted on position and filtered: {sortedbooks[["title", "price"]]}")
    topselling = books[books["position"] == 1]
    print(f"Top-selling book: {topselling}")
    print(f"Top-selling book title: {topselling["title"]}")
    print(f"Top-selling book title: {topselling["title"][0]}")

def reconcilation(issuedChecks, cashedChecks, duplicateRows):
    print(f"\n=== {reconcilation.__name__} ===")
    issued_types = [
        ("id", "i8"),
        ("payee", "U128"),
        ("amount", "f8"),
        ("issueDate", "U10")
    ]
    cashed_types = [
        ("id", "i8"),
        ("amount", "f8"),
        ("cashedDate", "U10")
    ]
    issued = numpy.loadtxt(Path(issuedChecks), delimiter=",", dtype=issued_types, skiprows=1)
    cashed = numpy.loadtxt(Path(cashedChecks), delimiter=",", dtype=cashed_types, skiprows=1)
    print(f"Issued checks: {issued}")
    print(f"Cashed checks: {cashed}")
    innerJoin = reconcile.rec_join("id", issued, cashed, jointype="inner")
    print(f"Inner joined dtype: {innerJoin.dtype}") # Numpy renames the 2 amount to amount1 and amount2
    print(f"Inner joined: {innerJoin}")
    print(f"Inner joined filtered: {innerJoin[["payee", "issueDate", "cashedDate"]]}")
    outstanding = [id for id in issued["id"] if id not in cashed["id"]]
    print(f"Outstanding checks: {outstanding}")
    duplicates = numpy.loadtxt(Path(duplicateRows), delimiter=",", dtype=issued_types, skiprows=1)
    rows = reconcile.find_duplicates(numpy.ma.asarray(duplicates))
    print(f"Duplicate rows: {rows}")
    unique = numpy.unique(duplicates, axis=0)
    print(f"Unique: {unique}")

def HierarchicalDataAnalysis(companies, prices):
    print(f"\n=== {HierarchicalDataAnalysis.__name__} ===")
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    weekdays_dtype = [(day, "f8") for day in weekdays]
    company_dtype = [("company", "U128"), ("sector", "U128")]
    portfolio_dtype = numpy.dtype(company_dtype + weekdays_dtype)
    portfolio = numpy.zeros(6, dtype=portfolio_dtype)
    companies = numpy.loadtxt(Path(companies), delimiter=",", dtype=company_dtype, skiprows=1) # Company column are sorted from A_Comp to F_Comp
    print(f"Companies: {companies}")
    #companies = companies.reshape(6)
    #print(f"Companies reshaped: {companies}") # XXX: I don't see any difference in shape
    portfolio[["company", "sector"]] = companies
    print(f"Portfolio: {portfolio}")
    prices_dtype = [("company", "U128"), ("day", "f8")]
    dayfiles = zip(weekdays, sorted(Path.cwd().glob(prices)))
    #print(f"dayfiles: {list(dayfiles)}") #This will break the following for-loop
    for day, price in dayfiles:
        portfolio[day] = numpy.loadtxt(price, delimiter=",", dtype=prices_dtype, skiprows=1)["day"] # Company column are sorted from A_Comp to F_Comp
    print(f"portfolio: {portfolio}")
    print(f"Prices for companies in the tech sector on Fri: {portfolio[portfolio["sector"] == "technology"]["Fri"]}")
    tech_mask = portfolio["sector"] == "technology"
    tech_sector_companies = portfolio[tech_mask]["company"]
    pricebars = plt.bar(x=tech_sector_companies, height=portfolio[tech_mask]["Fri"])
    pricebars[0].set_color("b")
    pricebars[1].set_color("r")
    plt.xlabel("Tech Companies")
    plt.ylabel("Friday Prices ($)")
    plt.title("Tech Share Valuation")
    plt.legend()
    #plt.show() This blocks
    plt.clf()
    plt.cla()
    
def Tensors():
    """
    Tensors in ML are multi-dimensional arrays used to and process data for NN
    Keras is a high-level NN API written in Python. It runs on top of deep-learning frameworks like TF. It wraps TF so that there is less code to write using Keras compared to using TF alone.
    https://www.tensorflow.org/hub - repository of pretrained models.
    https://ai.google.dev/edge/litert - light framework for IOT
    """
    print(f"\n=== {Tensors.__name__} ===")
    image_tensors = numpy.array([
        [255,0,255], # Row 1: white, black, white
        [0,255,0],  # Row 2: black, white, black
        [255,0,255] # Row 3: white, black, white
    ])
    array1 = numpy.array([[2.,4.,6.]])
    array2 = numpy.array([[1.],[3.],[5.]])
    result = tf.multiply(tf.convert_to_tensor(array1), tf.convert_to_tensor(array2))
    print("\ntf.multiply result: ")
    print(result)

@numpy.vectorize
def profit_with_bonus(first, last):
    #Bonus of 10% if profit is >= 1%
    return (last - first) * 1.1 if last >= first * 1.01 else last - first

def VectorOperations(data):
    print(f"\n=== {VectorOperations.__name__} ===")
    portfolio_dtype = [
        ("compay", "U128"),
        ("sector", "U128"),
        ("Mon", "f8"),
        ("Tue", "f8"),
        ("Wed", "f8"),
        ("Thu", "f8"),
        ("Fri", "f8"),
    ]
    portfolio = numpy.loadtxt(Path(data), delimiter=",", dtype=portfolio_dtype, skiprows=1)
    print(f"Portfolio: {portfolio}")
    # This will flip-flop between numpy and python lands
    bonuses = profit_with_bonus(portfolio["Mon"], portfolio["Fri"])
    print(f"Bonuses: {bonuses}")
    # This has the benefit of stayin in the numpy land and thus more optimized
    bonuses = numpy.where(
        portfolio["Fri"] >= portfolio["Mon"] * 1.01,
        (portfolio["Fri"] - portfolio["Mon"]) * 1.1,
        portfolio["Fri"] - portfolio["Mon"]
    )
    print(f"Bonuses: {bonuses}")

def VectorProperties():
    """
    https://realpython.com/chromadb-vector-database/
    """
    print(f"\n=== {VectorProperties.__name__} ===")
    v1 = numpy.array([1, 0])
    v2 = numpy.array([0, 1])
    v3 = numpy.array([numpy.sqrt(2), numpy.sqrt(2)])
    """
    Dimension: The dimension of a vector is the number of elements that it contains. In the example above, vector1 and vector2 are both two-dimensional since they each have two elements. 
    You can only visualize vectors with three dimensions or less, but generally, vectors can have any number of dimensions. 
    Vectors that encode words and text tend to have hundreds or thousands of dimensions.
    Using numpy, .shape gives the dimension
    """
    print(f"Dimension: {v1.shape}")
    """
    Magnitude: The magnitude of a vector is a non-negative number that represents the vector’s size or length. You can also refer to the magnitude of a vector as the norm, and you can denote it with ||v|| or |v|. 
    There are many different definitions of magnitude or norm, but the most common is the Euclidean norm or 2-norm. 2 ways to compute:
    (1) Euclidean norm
    (2) np.linalg.norm(), a NumPy function that computes the Euclidean norm
    """
    print(f"Euclidean norm: {numpy.sqrt(numpy.sum(v1**2))}, {numpy.linalg.norm(v1)}")
    assert numpy.sqrt(numpy.sum(v3**2)) == numpy.linalg.norm(v3)
    """
    Dot product (scalar product): The dot product of two vectors, u and v, is a number given by u ⋅ v = ||u|| ||v|| cos(θ), where θ is the angle between the two vectors. 
    Another way to compute the dot product is to do an element-wise multiplication of u and v and sum the results. 
    The dot product is one of the most important and widely used vector operations because it measures the similarity between two vectors. 2 ways to compute:
    """
    print(f"Dot product (v1.v3): {numpy.sum(v1 * v3)}, {v1 @ v3}")
    assert numpy.sum(v1 * v3) == v1 @ v3 > 0
    assert numpy.sum(v2 * v3) == v2 @ v3 > 0
    assert 0 == v1 @ v2 # Orthogonal OR Unrelated

def Polynomial():
    print(f"\n=== {Polynomial.__name__} ===")
    x = numpy.array([66, 5, 15, 2, 500])
    c = numpy.array([3000, 200 , -50, 5000, 100])
    # @ operator = numpy.matmul (matrix multiplication)
    print(f"x @ c: {x @ c}, numpy.dot(x, c): {numpy.dot(x, c)}")
    x = numpy.array([[66, 5, 15, 2, 500], 
              [21, 3, 50, 1, 100]])
    print(f"x @ c: {x @ c}, numpy.dot(x, c): {numpy.dot(x, c)}")

def kMin(n): #N minimum (index) values in a numpy array
    print(f"\n=== {kMin.__name__} ===")
    # https://stackoverflow.com/questions/16817948/i-have-need-the-n-minimum-index-values-in-a-numpy-array
    arr = numpy.array([1, 4, 2, 5, 3])
    indices = arr.argsort()[:n] # Sort the array
    print(f"arr: {arr}, tmp: {indices}")

def VectorSlicing():
    print(f"\n=== {VectorSlicing.__name__} ===")
    #vector slicing operations
    a = numpy.arange(10)
    print(f"a         = {a}")

    #access 5 consecutive elements (start:stop:step)
    c = a[2:7:1];     print("a[2:7:1] = ", c)

    # access 3 elements separated by two 
    c = a[2:7:2];     print("a[2:7:2] = ", c)

    # access all elements index 3 and above
    c = a[3:];        print("a[3:]    = ", c)

    # access all elements below index 3
    c = a[:3];        print("a[:3]    = ", c)

    # access all elements
    c = a[:];         print("a[:]     = ", c)

def RandomTests():
    print(f"\n=== {RandomTests.__name__} ===")
    data = numpy.arange(0, 10, 1)
    print(f"data: {data}")
    print(f"data (reversed): {data[::-1]}")

    data = rng.standard_normal(100)
    print(f"data: {data}")
    indices = data.argsort()
    print(f"Sorted indices: {indices}")
    print(f"Sorted data: {data[indices]}")
    print(f"min: [{numpy.argmin(data)}] {numpy.min(data)}, max: [{numpy.argmax(data)}] {numpy.max(data)}")
    top10_indices = indices[-10:]
    print(f"Top-10 indices: {top10_indices}")
    print(f"Top-10 data: {data[top10_indices]}")
    top10_descending_indices = top10_indices[::-1]
    print(f"Top-10 indices (descending): {top10_descending_indices}")
    print(f"Top-10 data  (descending): {data[top10_descending_indices]}")

def NoisySineWave(samples: int):
    print(f"\n=== {NoisySineWave.__name__} ===")
    # Generate some random samples
    #numpy.random.seed(1234)
    x_values = numpy.random.uniform(low=0, high=(2 * math.pi), size=samples)
    plt.plot(x_values)
    plt.title("Noisy Sine Wave X values")
    plt.show()
    # Create a noisy sinewave with these values
    y_values = numpy.sin(x_values) + (0.1 * numpy.random.randn(x_values.shape[0]))
    plt.plot(x_values, y_values, '.')
    plt.title("Noisy Sine Wave")
    plt.show()

def ShapeTests():
    print(f"\n=== {ShapeTests.__name__} ===")
    a = numpy.ndarray((2,2))
    assert (2,2) == a.shape
    print(f"a.shape: {a.shape}")
    a = a.reshape((1,)+ a.shape)
    assert (1,2,2) == a.shape
    print(f"a.shape: {a.shape}")

if __name__ == "__main__":
    InitializeGPU()
    oneDArray(10)
    twoDArray(30)
    Matrix()
    multiDArray(3,4,5)
    AdvancedIndexing()
    Broadcasting()
    ConcatenateSliceObjects()
    csvArray(3,2,3, "data/file?.csv")
    """
    https://stackoverflow.com/questions/17079279/how-is-axis-indexed-in-numpys-array
    The axis number of the dimension is the index of that dimension within the array's shape.
    For example, if a 2D array a has shape (5,6), then you can access a[0,0] up to a[4,5]. 
    Axis 0 is thus the first dimension (the "rows"), and axis 1 is the second dimension (the "columns"). 
    In higher dimensions, where "row" and "column" stop really making sense, try to think of the axes in terms of the shapes and indices involved.

    https://stackoverflow.com/questions/46855793/understanding-axes-in-numpy
    In numpy, axis ordering follows zyx convention, instead of the usual (and maybe more intuitive) xyz.

    https://stackoverflow.com/questions/79429728/python-multidimension-array-axis-and-index-to-load-csv-file-into
    """
    csvArrayInsert(4,2,3,"data/file?.csv",2,0,1,"data/file4_extra_row.csv")
    structuredArray()
    reconcilation("data/issued_checks.csv", "data/cashed_checks.csv", "data/issued_dupe.csv")
    HierarchicalDataAnalysis("data/companies.csv", "data/prices-?.csv")
    VectorOperations("data/full_portfolio.csv")
    Tensors()
    VectorProperties()
    VectorSlicing()
    Polynomial()
    kMin(3)
    RandomTests()
    ShapeTests()
    NoisySineWave(1024)