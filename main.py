import numpy
from utils.UniVariatePlot import UniVariatePlot
portnames = ["PAN", "AMS", "CAS", "NYC", "HEL"]
D = [
        [0,8943,8019,3652,10545],
        [8943,0,2619,6317,2078],
        [8019,2619,0,5836,4939],
        [3652,6317,5836,0,7825],
        [10545,2078,4939,7825,0]
    ]

# https://timeforchange.org/co2-emissions-shipping-goods
# assume 20g per km per metric ton (of pineapples)

co2 = 0.020
# DATA BLOCK ENDS

# these variables are initialised to nonsensical values
# your program should determine the correct values for them
smallest = 1000000
bestroute = [0, 0, 0, 0, 0]
 
def permutations(route, ports):
    #print(f"route so far: {route}, ports: {ports}")
    if not ports or len(ports) == 0:
        global co2, smallest, bestroute
        print(' '.join([portnames[i] for i in route]))
        #for i in range(0, len(route) - 1):
        #    emission += D[route[i]][route[i+1]]
        emission = co2 * sum(D[i][j] for i, j in zip(route[:-1], route[1:]))
        if emission < smallest:
            smallest = emission
            bestroute = list(route)
    else:
        for i in ports:
            #print(f"Processing port {i} of {ports}, Route: {route}")
            #if i not in route:
            newRoute = list(route)
            newRoute.append(i)
            #print(f"newRoute: {newRoute}")
            j = ports.index(i)
            #print(f"Remaining ports: {ports[:j]+ports[j+1:]}")
            permutations(newRoute, ports[:j]+ports[j+1:])
    # Print the port names in route when the recursion terminates

def polynomial():
    print(f"\n=== {polynomial.__name__} ===")
    x = numpy.array([66, 5, 15, 2, 500])
    c = numpy.array([3000, 200 , -50, 5000, 100])
    # @ operator = numpy.matmul (matrix multiplication)
    print(f"x @ c: {x @ c}, numpy.dot(x, c): {numpy.dot(x, c)}")
    x = numpy.array([[66, 5, 15, 2, 500], 
              [21, 3, 50, 1, 100]])
    print(f"x @ c: {x @ c}, numpy.dot(x, c): {numpy.dot(x, c)}")

def kmin(n): #N minimum (index) values in a numpy array
    print(f"\n=== {kmin.__name__} ===")
    # https://stackoverflow.com/questions/16817948/i-have-need-the-n-minimum-index-values-in-a-numpy-array
    arr = numpy.array([1, 4, 2, 5, 3])
    indices = arr.argsort()[:n] # Sort the array
    print(f"arr: {arr}, tmp: {indices}")

def testZip():
    print(f"\n=== {testZip.__name__} ===")
    a = [1,2,3,4,5]
    b = [9,8,3,5,6]
    for i, j in zip(a,b):
        print(f"[{i}, {j}]")
    c = [abs(i- j) for i, j in zip(a, b)]
    print(f"Manhattan distance: {c}")
    d = sum([abs(i- j) for i, j in zip(a, b)])
    print(f"Manhattan distance: {d}")

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

def primary_diagonal(matrix):
    sum = 0
    for i in range(len(matrix)):
        sum += matrix[i][i]
    return sum

def secondary_left_diagonal(matrix):
    sum = 0
    for i in range(len(matrix)):
        sum += matrix[i][len(matrix) - i - 1]
    return sum

def matrix_sums():
    print(f"\n=== {matrix_sums.__name__} ===")
    matrix = [
        [1,2,3,4],
        [1,2,3,4],
        [1,2,3,4],
        [1,2,3,4]
    ]
    row_totals = [ sum(x) for x in matrix ]
    col_totals = [ sum(x) for x in zip(*matrix) ]
    diag1 = primary_diagonal(matrix)
    diag2 = secondary_left_diagonal(matrix)
    print(f"rows: {row_totals}, cols: {col_totals}, diag1: {diag1}, diag2: {diag2}")

def sort_dict_by_tuple_values():
    result = {
        "a": (0,10),
        "b": (10,5),
        "c": (5,5),
        "d": (7,8),
        "e": (8,7),
        "f": (6,6),
        "g": (5,6),
    }
    result = {k: v for k, v in sorted(result.items(), key=lambda item: (item[1][0], -item[1][1]))}
    for k,v in result.items():
        print(f"{k}: {v}")

def univariate_plot():
    x_train = numpy.array([1.0, 2.0])
    y_train = numpy.array([300.0, 500.0])
    UniVariatePlot(x_train, y_train, [], 0.0, "Housing Prices", 'Price (in 1000s of dollars)', 'Size (1000 sqft)')

def main():
    sort_dict_by_tuple_values()
    permutations([0], list(range(1, len(portnames)))) # This will start the recursion with 0 ("PAN") as the first stop
    print(' '.join([portnames[i] for i in bestroute]) + " %.1f kg" % smallest)
    polynomial()
    kmin(3)
    testZip()
    VectorSlicing()
    Matrix()
    matrix_sums()
    #univariate_plot() This blocks

if __name__ == "__main__":
    main()