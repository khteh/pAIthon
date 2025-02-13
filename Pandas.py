import pandas as pd
import numpy
def numpy_dataframe():
    print(f"=== {numpy_dataframe.__name__} ===")
    d = {'x': range(10), 'y': numpy.arange(1,20, 2), 'z': 100}
    indices = range(100,110)
    data = pd.DataFrame(d, index=indices, columns=['z','y','x'])
    print(f"data ({id(data)}), ndim: {data.ndim}, size: {data.size}, shape: {data.shape}")
    print("Memory usage:")
    print(data.memory_usage())
    print("\nHead:")
    print(data.head())
    print("\nTail:")
    print(data.tail())

def DataFrameAttributes():
    print(f"\n=== {DataFrameAttributes.__name__} ===")
    data = {
        'Name': ['Xavier', 'Ann', 'Jana', 'Yi', 'Robin', 'Amal', 'Nori'],
        'City': ['Mexico City', 'Toronto', 'Prague', 'Shanghai', 'Manchester', 'Cairo', 'Osaka'],
        'Age': [41, 28, 33, 35, 37, 38, 45],
        'Python': [88.0, 79.0, 81.0, 80.0, 68.0, 61.0, 84.0]
    }
    indices = range(101,108)
    data = pd.DataFrame(data, index = indices)
    print(f"data ({id(data)}), ndim: {data.ndim}, size: {data.size}, shape: {data.shape}")
    print("Memory usage:")
    print(data.memory_usage())
    print("\nData types:")
    print(data.dtypes)
    print("\nIndex / row labels:")
    print(data.index)
    print("\nColumn labels:")
    print(data.columns)
    print("\nHead:")
    print(data.head())
    print("\nTail:")
    print(data.tail())

def DataFrameAccess():
    print(f"\n=== {DataFrameAccess.__name__} ===")
    data = {
        'Name': ['Xavier', 'Ann', 'Jana', 'Yi', 'Robin', 'Amal', 'Nori'],
        'City': ['Mexico City', 'Toronto', 'Prague', 'Shanghai', 'Manchester', 'Cairo', 'Osaka'],
        'Age': [41, 28, 33, 35, 37, 38, 45],
        'Python': [88.0, 79.0, 81.0, 80.0, 68.0, 61.0, 84.0]
    }
    indices = range(101,108)
    data = pd.DataFrame(data, index = indices)
    print(f"data ({id(data)}), ndim: {data.ndim}, size: {data.size}, shape: {data.shape}")
    print("\nCities:")
    print(data['City']) # pamdas.core.series.Series data type
    print("\nRow 104 accessed by index key / row label:")
    print(data.loc[104])# pamdas.core.series.Series data type
    print("\nRow 104 accessed by 0-based index counter:")
    print(data.iloc[3])# pamdas.core.series.Series data type
    print(f"City at row 103: {data.City[103]}") # Series object permits usage of direct indexing
    print("\nValues:")
    print(data.values)
    print("\nAge and Score accessed by column index key / row label:")
    print(data.loc[:, ['Age', 'Python']])
    print("\nAge and Score accessed by 0-based column index counters:")
    print(data.iloc[:, [2,3]])
    print("Row 104 City accessed using row/column labels:")
    print(data.loc[104, 'City'])
    print(data.at[104, 'City'])
    print("Row 104 City accessed using index counter:")
    print(data.iat[3, 1])
    print("\nModify the scores with evenly-spaced values between 50 and 90:")
    data.iloc[:, -1] = numpy.linspace(50,90,len(data))
    print(data)

def DataFrameModifications():
    print(f"\n=== {DataFrameModifications.__name__} ===")
    data = {
        'Name': ['Xavier', 'Ann', 'Jana', 'Yi', 'Robin', 'Amal', 'Nori'],
        'City': ['Mexico City', 'Toronto', 'Prague', 'Shanghai', 'Manchester', 'Cairo', 'Osaka'],
        'Age': [41, 28, 33, 35, 37, 38, 45],
        'Python': [88.0, 79.0, 81.0, 80.0, 68.0, 61.0, 84.0]
    }
    indices = range(101,108)
    data = pd.DataFrame(data, index = indices)
    print(f"data ({id(data)}), ndim: {data.ndim}, size: {data.size}, shape: {data.shape}")
    john = pd.Series(data=['John', 'Boston', 39, 79.5], index=data.columns, name=data.index[-1] + 1)
    data.loc[data.index[-1] + 1] = john
    print("\nAfter appending new row:")
    print(data)
    print("\nAfter deleting a row:")
    data.drop(labels=[108], inplace=True)
    print(data)
    print("\nAdding a column at the end:")
    data["C++"] = [78.0, 69.0, 85.0, 83.0, 72.0, 68.0, 85.0]
    print(data)
    print("\nAdding a column at specific location:")
    data.insert(loc=3, column="C#",value= [78.0, 69.0, 85.0, 83.0, 72.0, 68.0, 85.0])
    print(data)
    print("\nDelete a column using python dectionary deletion:")
    del data['C#']
    print(data)
    print("\nDelete a column using pop():")
    score1 = data.pop('C++')
    print(score1)
    print(data)
    print("\nDelete a column using drop():")
    data.drop(labels=['Age'], axis=1, inplace=True)
    print(data)

def DataFrameArithmetic():
    print(f"\n=== {DataFrameArithmetic.__name__} ===")
    data = {
        'Name': ['Xavier', 'Ann', 'Jana', 'Yi', 'Robin', 'Amal', 'Nori'],
        'City': ['Mexico City', 'Toronto', 'Prague', 'Shanghai', 'Manchester', 'Cairo', 'Osaka'],
        'Age': [41, 28, 33, 35, 37, 38, 45],
        'Python': [88.0, 79.0, 81.0, 80.0, 68.0, 61.0, 84.0]
    }
    indices = range(101,108)
    data = pd.DataFrame(data, index = indices)
    print(f"data ({id(data)}), ndim: {data.ndim}, size: {data.size}, shape: {data.shape}")
    print("\nAdding a column at the end:")
    data["C++"] = [78.0, 69.0, 85.0, 83.0, 72.0, 68.0, 85.0]
    print(data)
    print("\nAdding a column at specific location:")
    data.insert(loc=3, column="C#",value= [78.0, 69.0, 85.0, 83.0, 72.0, 68.0, 85.0])
    print(data)
    weights = pd.Series(data=[0.3,0.4,0.3],index=['C#','Python','C++'])
    print("\nWeights:")
    print(weights)
    print("Weight.index:")
    print(weights.index)
    print("\nWeighted Scores:")
    print(data[weights.index] * weights)
    print("\nTotal score of each skill:")
    data.loc[data.index[-1] + 1] = numpy.sum(data[weights.index] * weights)
    #print(numpy.sum(data[weights.index] * weights))
    print(data)
    print("\nTotal score of each individual:")
    data['Total'] = numpy.sum(data[weights.index] * weights, axis=1)
    print(data)
#####
numpy_dataframe()
DataFrameAttributes()
DataFrameAccess()
DataFrameModifications()
DataFrameArithmetic()