import numpy
from pathlib import Path
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
    print(f"\n=== {sort_dict_by_tuple_values.__name__} ===")
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
    print(f"\n=== {univariate_plot.__name__} ===")
    x_train = numpy.array([1.0, 2.0])
    y_train = numpy.array([300.0, 500.0])
    UniVariatePlot(x_train, y_train, [], 0.0, "Housing Prices", 'Price (in 1000s of dollars)', 'Size (1000 sqft)')

def PathlibMkdirTest():
    print(f"\n=== {PathlibMkdirTest.__name__} ===")
    Path("output/SineWaveGAN").mkdir(parents=True, exist_ok=True)
    Path("output/SineWaveGAN").is_dir()

def TupleArithmetic():
    print(f"\n=== {TupleArithmetic.__name__} ===")
    t1 = (1,2,3)
    t2 = t1 + (4,)
    assert (1,2,3,4) == t2
    print(t2)

def ArrayExtension():
    print(f"\n=== {ArrayExtension.__name__} ===")
    array = ['accuracy'] * 10
    assert 10 == len(array)
    print(f"array: {array}")

def CountSubStrings(src, substr, repetition):
    print(f"\n=== {CountSubStrings.__name__} ===")
    src *= repetition
    print(f"'{substr}' appears in `{src}` {src.count(substr)} times")

if __name__ == "__main__":
    sort_dict_by_tuple_values()
    permutations([0], list(range(1, len(portnames)))) # This will start the recursion with 0 ("PAN") as the first stop
    print(' '.join([portnames[i] for i in bestroute]) + " %.1f kg" % smallest)
    testZip()
    matrix_sums()
    PathlibMkdirTest()
    #univariate_plot() This blocks
    TupleArithmetic()
    ArrayExtension()
    CountSubStrings("banana", "an", 4)