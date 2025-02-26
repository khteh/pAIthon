import numpy, tensorflow as tf
from pathlib import Path
import numpy.lib.recfunctions as reconcile
import matplotlib.pyplot as plt

# Hide GPU from visible devices
def InitializeGPU():
    """
    2024-12-17 12:39:33.030218: I external/local_xla/xla/stream_executor/cuda/cuda_driver.cc:1193] failed to allocate 2.2KiB (2304 bytes) from device: RESOURCE_EXHAUSTED: : CUDA_ERROR_OUT_OF_MEMORY: out of memory
    https://stackoverflow.com/questions/39465503/cuda-error-out-of-memory-in-tensorflow
    https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
    https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth
    https://www.tensorflow.org/guide/gpu
    """
    #tf.config.set_visible_devices([], 'GPU')
    #tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_physical_devices('GPU')
    print(f"{len(gpus)} GPUs available")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# https://numpy.org/doc/stable/user/basics.indexing.html
def oneDArray(n: int):
    print(f"=== {oneDArray.__name__} ===")
    one = numpy.array(range(n))
    print(f"{one} shape: {one.shape}\n")
    one = numpy.arange(n)
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
    aggregate0 = numpy.sum(numbers, axis=0) # Each column collapsed into a single row
    aggregate1 = numpy.sum(numbers, axis=1) # Each row collapsed into a single column
    print(f"Aggregates axis-0: {aggregate0}, axis-1: {aggregate1}")

def multiDArray(i:int, j: int, k:int):
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
    # z-axis consists of 2 elements; y-axis consists of 3 elements; x-axis consists of 1 element:
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
    #plt.show() This blocks
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

if __name__ == "__main__":
    InitializeGPU()
    oneDArray(10)
    twoDArray(30)
    multiDArray(3,4,5)
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