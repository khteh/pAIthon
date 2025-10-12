import numpy, pandas as pd
from pathlib import Path
from utils.UniVariatePlot import UniVariatePlot
from utils.CosineSimilarity import cosine_similarity

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

def DeleteCharFromString(word: str):
    '''
    Input:
        word: the string/word for which you will generate all possible words 
                in the vocabulary which have 1 missing character
    Output:
        delete_l: a list of all possible strings obtained by deleting 1 character from word
    '''    
    print(f"\n=== {DeleteCharFromString.__name__} ===")
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    result = [L + R[1:] for L, R in splits if R] # This always removes the first character, [0] from the second concatenated string, R
    print(f"result: {result}")

def InsertCharString(word: str):
    '''
    Input:
        word: the input string/word 
    Output:
        inserts: a set of all possible strings with one new letter inserted at every offset
    ''' 
    print(f"\n=== {InsertCharString.__name__} ===")
    chars = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    result = []
    for L, R in splits: # This always replaces the first character, [0] from the second concatenated string, R
        for c in chars:
            result.append(L + c + R)
    print(f"result: {len(result)} {result}")

def SwitchLetters(word: str):
    '''
    Input:
        word: input string
     Output:
        switches: a list of all possible strings with one adjacent charater switched
    '''
    print(f"\n=== {SwitchLetters.__name__} ===")
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    """
    ince, ncie, niec
    [('', 'nice'), ('n', 'ice), ('ni', 'ce'), ('nic', 'e'), 'nice', '')]
    ('', 'nice') : 'ince'
    ('n', 'ice) : 'n' + 'cie'
    ('ni', 'ce') : 'ni' + 'ec'
    """
    result = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) >= 2]
    print(f"result: {result}")

def ReplaceCharsPermutation(word: str):
    '''
    Input:
        word: the input string/word 
    Output:
        replaces: a list of all possible strings where we replaced one letter from the original word. 
    '''
    print(f"\n=== {ReplaceCharsPermutation.__name__} ===")
    chars = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    result = set()
    for L, R in splits: # This always replaces the first character, [0] from the second concatenated string, R
        if R:
            for c in chars:
                tmp = L + c + R[1:]
                if tmp != word:
                    result.add(tmp)
    print(f"result: {len(result)} {result}")
    return result

def min_edit_distance(source, target, ins_cost = 1, del_cost = 1, rep_cost = 2):
    '''
    Input: 
        source: a string corresponding to the string you are starting with
        target: a string corresponding to the string you want to end with
        ins_cost: an integer setting the insert cost
        del_cost: an integer setting the delete cost
        rep_cost: an integer setting the replace cost
    Output:
        D: a matrix of len(source)+1 by len(target)+1 containing minimum edit distances
        med: the minimum edit distance (med) required to convert the source string to the target
    '''
    print(f"\n=== {min_edit_distance.__name__} ===")
    # use deletion and insert cost as  1
    m = len(source) 
    n = len(target) 
    #initialize cost matrix with zeros and dimensions (m+1,n+1) 
    D = numpy.zeros((m+1, n+1), dtype=int) 
    
    # Fill in column 0, from row 1 to row m, both inclusive
    for row in range(1, m + 1): # Replace None with the proper range
        D[row,0] = D[row-1,0] + del_cost
        
    # Fill in row 0, for all columns from 1 to n, both inclusive
    for col in range(1, n + 1): # Replace None with the proper range
        D[0,col] = D[0, col-1] + ins_cost
        
    # Loop through row 1 to row m, both inclusive
    for i in range(1,m+1):
        
        # Loop through column 1 to column n, both inclusive
        for j in range(1,n+1):
            
            # Intialize r_cost to the 'replace' cost that is passed into this function
            r_cost = 0 if source[i-1] == target[j-1] else rep_cost
            
            # Update the cost at row, col based on previous entries in the cost matrix
            # Refer to the equation calculate for D[i,j] (the minimum of three calculated costs)
            D[i,j] = min(D[i-1,j] + del_cost, D[i,j-1] + ins_cost, D[i-1,j-1] + r_cost)
            
    # Set the minimum edit distance with the cost found at row m, column n 
    med = D[m,n]
    return D, med

def min_edit_distance_tests():
    print(f"\n=== {min_edit_distance_tests.__name__} ===")
    source =  'play'
    target = 'stay'
    matrix, min_edits = min_edit_distance(source, target)
    assert 4 == min_edits
    print("minimum edits: ",min_edits, "\n")
    idx = list('#' + source)
    cols = list('#' + target)
    df = pd.DataFrame(matrix, index=idx, columns= cols)
    print(df)

    source =  'eer'
    target = 'near'
    matrix, min_edits = min_edit_distance(source, target)
    assert 3 == min_edits
    print("minimum edits: ",min_edits, "\n")
    idx = list(source)
    idx.insert(0, '#')
    cols = list(target)
    cols.insert(0, '#')
    df = pd.DataFrame(matrix, index=idx, columns= cols)
    print(df)

def CosineSimilarityTests():
    print(f"\n=== {CosineSimilarityTests.__name__} ===")
    usa = numpy.array([5,6])
    washington = numpy.array([10,5])
    turkey = numpy.array([3,1])
    russia = numpy.array([5,5])
    japan = numpy.array([4,3])
    ankara = numpy.array([9,1])
    # usa - washington = ? - ankara
    # ? = usa - washington + ankara
    country = usa - washington + ankara
    print(f"embedding of country of ankara: {country}")
    for c in [turkey, russia, japan]:
        print(f"{c}: {cosine_similarity(country, c)}")

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
    DeleteCharFromString("banana")
    InsertCharString("at")
    SwitchLetters("nice")
    strings = set(['aan', 'ban', 'caa', 'cab', 'cac', 'cad', 'cae', 'caf', 'cag', 'cah', 'cai', 'caj', 'cak', 'cal', 'cam', 'cao', 'cap', 'caq', 'car', 'cas', 'cat', 'cau', 'cav', 'caw', 'cax', 'cay', 'caz', 'cbn', 'ccn', 'cdn', 'cen', 'cfn', 'cgn', 'chn', 'cin', 'cjn', 'ckn', 'cln', 'cmn', 'cnn', 'con', 'cpn', 'cqn', 'crn', 'csn', 'ctn', 'cun', 'cvn', 'cwn', 'cxn', 'cyn', 'czn', 'dan', 'ean', 'fan', 'gan', 'han', 'ian', 'jan', 'kan', 'lan', 'man', 'nan', 'oan', 'pan', 'qan', 'ran', 'san', 'tan', 'uan', 'van', 'wan', 'xan', 'yan', 'zan'])
    result = ReplaceCharsPermutation("can")
    print(f"\nstrings: {len(strings)}, difference: {result - strings}")
    min_edit_distance_tests()
    CosineSimilarityTests()