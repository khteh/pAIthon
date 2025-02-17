import pandas as pd
import numpy
from collections import namedtuple
"""
DataFrame objects are collections of Series objects.
Each item in a Series has an index.
"""
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
        'Python': [88.5, 79.0, 81.5, 80.0, 68.5, 61.0, 84.5]
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
        'Python': [88.5, 79.0, 81.5, 80.0, 68.5, 61.0, 84.5]
    }
    indices = range(101,108)
    data = pd.DataFrame(data, index = indices)
    print(f"data ({id(data)}), ndim: {data.ndim}, size: {data.size}, shape: {data.shape}")
    print("\nCities:")
    print(data['City']) # pandas.core.series.Series data type. Each item in the Series has an index.
    print("\nRow 104 accessed by index key / row label:")
    print(data.loc[104])# pandas.core.series.Series data type. Each item in the Series has an index.
    print("\nRow 104 accessed by 0-based index counter:")
    print(data.iloc[3])# pandas.core.series.Series data type. Each item in the Series has an index.
    print(f"City at row 103: {data.City[103]}") # Series object permits usage of direct indexing
    print("\nValues:")
    print(data.values)
    print("\nAge and Score accessed by column index key / row label:")
    print(data.loc[:, ['Age', 'Python']])
    print("\nAge and Score accessed by column index key / row label [103:105]:")
    print(data.loc[103:105, ['Age', 'Python']])
    print("\nRow 105 with all columns:")
    print(data.loc[105, :])
    print("\nUsing .loc with a filter:")
    print(data.loc[data.Name == "Robin"])
    print("\nAge and Score accessed by 0-based column index counters:")
    print(data.iloc[:, [2,3]])
    print("Row 104 City accessed using row/column labels:")
    print(data.loc[104, 'City'])
    print(data.at[104, 'City'])
    print("Row 104 City accessed using index counter:")
    print(data.iat[3, 1])
    print("\nData before modification:")
    print(data)
    print("\nModify the row[1:5) Python column (axis-1) with evenly-spaced values between 0 and 100:")
    data.iloc[1:5, -1] = numpy.linspace(0,100,4)
    print(data)
    print("\nModify the Python column (axis-1) with evenly-spaced values between 50 and 90:")
    data.iloc[:, -1] = numpy.linspace(50,90,len(data))
    print(data)

def DataFrameModifications():
    print(f"\n=== {DataFrameModifications.__name__} ===")
    data = {
        'Name': ['Xavier', 'Ann', 'Jana', 'Yi', 'Robin', 'Amal', 'Nori'],
        'City': ['Mexico City', 'Toronto', 'Prague', 'Shanghai', 'Manchester', 'Cairo', 'Osaka'],
        'Age': [41, 28, 33, 35, 37, 38, 45],
        'Python': [88.5, 79.0, 81.5, 80.0, 68.5, 61.0, 84.5]
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
    data["C++"] = [78.5, 69.0, 85.5, 83.0, 72.5, 68.0, 85.5]
    print(data)
    print("\nAdding a column at specific location:")
    data.insert(loc=3, column="C#",value= [75.0, 69.0, 85.0, 81.5, 73.5, 68.5, 83.5])
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
        'Python': [88.5, 79.0, 81.5, 80.0, 68.5, 61.0, 84.5]
    }
    indices = range(101,108)
    data = pd.DataFrame(data, index = indices)
    print(f"data ({id(data)}), ndim: {data.ndim}, size: {data.size}, shape: {data.shape}")
    data["C++"] = [78.5, 69.0, 85.5, 83.0, 72.5, 68.0, 85.5]
    data.insert(loc=3, column="C#",value= [75.0, 83.5, 69.5, 81.5, 73.5, 68.5, 84.5])
    print(data)
    weights = pd.Series(data=[0.3,0.4,0.3],index=['C#','Python','C++'])
    print("\nWeights:")
    print(weights)
    print("Weight.index:")
    print(weights.index)
    print("\nWeighted Scores:")
    print(data[weights.index] * weights)
    print("\nWeighted Scores sum:")
    print(numpy.sum(data[weights.index] * weights))
    print("\nTotal score of each individual:")
    data['Total'] = numpy.sum(data[weights.index] * weights, axis=1)
    print(data)
    print("\nTotal score of each individual (sorted):")
    print(f"data.index[-1]: {data.index[-1]}")
    data = data.sort_values(by=['C++', 'C#'], ascending=[False, False]) # Tie-breaks in C++ scores between Jana and Nori
    print(f"data.index[-1]: {data.index[-1]}")
    # Add a row for the totals of each skill. Adding here will disrupt the data table due to reordering of data.index[-1]
    #data.loc[data.index[-1] + 1] = numpy.sum(data[weights.index] * weights) 
    print(data)

def DataFrameFiltering():
    print(f"\n=== {DataFrameFiltering.__name__} ===")
    data = {
        'Name': ['Xavier', 'Ann', 'Jana', 'Yi', 'Robin', 'Amal', 'Nori'],
        'City': ['Mexico City', 'Toronto', 'Prague', 'Shanghai', 'Manchester', 'Cairo', 'Osaka'],
        'Age': [41, 28, 33, 35, 37, 38, 45],
        'Python': [88.5, 79.0, 81.5, 80.0, 68.5, 61.0, 84.5]
    }
    indices = range(101,108)
    data = pd.DataFrame(data, index = indices)
    print(f"data ({id(data)}), ndim: {data.ndim}, size: {data.size}, shape: {data.shape}")
    data["C++"] = [78.5, 69.0, 85.5, 83.0, 72.5, 68.0, 85.5]
    data.insert(loc=3, column="C#",value= [75.0, 83.5, 69.5, 81.5, 73.5, 68.5, 84.5])
    print(data)
    filter = (data['C++'] >= 80) & (data['C#'] >= 80)
    print("\nFiltered data:")
    print(data[filter])
    print("\nFiltered data using where and setting default value for unmet condition:")
    data['Python'] = data['Python'].where(cond=data['Python'] >= 80, other=0.0)
    print(data)
    print("\nFiltered data using regex:")
    data = data.filter(regex=r"^C(?:\+\+|#)?$")
    print(data)

def DataFrameStatistics():
    print(f"\n=== {DataFrameStatistics.__name__} ===")
    data = {
        'Name': ['Xavier', 'Ann', 'Jana', 'Yi', 'Robin', 'Amal', 'Nori'],
        'City': ['Mexico City', 'Toronto', 'Prague', 'Shanghai', 'Manchester', 'Cairo', 'Osaka'],
        'Age': [41, 28, 33, 35, 37, 38, 45],
        'Python': [88.5, 79.0, 81.5, 80.0, 68.5, 61.0, 84.5]
    }
    indices = range(101,108)
    data = pd.DataFrame(data, index = indices)
    print(f"data ({id(data)}), ndim: {data.ndim}, size: {data.size}, shape: {data.shape}")
    data["C++"] = [78.5, 69.0, 85.5, 83.0, 72.5, 68.0, 85.5]
    data.insert(loc=3, column="C#",value= [75.0, 83.5, 69.5, 81.5, 73.5, 68.5, 84.5])
    print(data)
    print(data.describe())

def DataFrameMissingDataHandling():
    print(f"\n=== {DataFrameMissingDataHandling.__name__} ===")
    data = pd.DataFrame({'x': [1,2,numpy.nan, 4]})
    print(f"DataFrame with a missing value replaced with {numpy.nan}:")
    print(data)
    print(data.describe())
    print(f"DataFrame with a missing value in a row dropped:")
    data1 = data.dropna()
    print(data1)
    print(data1.describe())
    print(f"DataFrame with a missing value in a column dropped:")
    data1 = data.dropna(axis=1)
    print(data1)
    #print(data1.describe())
    print(f"DataFrame with a missing value replaced with 0:")
    data1 = data.fillna(value=0)
    print(data1)
    print(data1.describe())
    print(f"DataFrame with a missing value replaced with previous value:")
    data1 = data.ffill() # forward-fill
    print(data1)
    print(data1.describe())
    print(f"DataFrame with a missing value replaced with next value:")
    data1 = data.bfill() # backward-fill
    print(data1)
    print(data1.describe())
    print(f"DataFrame with a missing value replaced with linear interpolation:")
    data1 = data.interpolate()
    print(data1)
    print(data1.describe())

def DataFrameIteration():
    print(f"\n=== {DataFrameIteration.__name__} ===")
    data = {
        'Name': ['Xavier', 'Ann', 'Jana', 'Yi', 'Robin', 'Amal', 'Nori'],
        'City': ['Mexico City', 'Toronto', 'Prague', 'Shanghai', 'Manchester', 'Cairo', 'Osaka'],
        'Age': [41, 28, 33, 35, 37, 38, 45],
        'Python': [88.5, 79.0, 81.5, 80.0, 68.5, 61.0, 84.5]
    }
    indices = range(101,108)
    data = pd.DataFrame(data, index = indices)
    print(f"data ({id(data)}), ndim: {data.ndim}, size: {data.size}, shape: {data.shape}")
    data["C++"] = [78.5, 69.0, 85.5, 83.0, 72.5, 68.0, 85.5]
    data.insert(loc=3, column="C#",value= [75.0, 83.5, 69.5, 81.5, 73.5, 68.5, 84.5])
    print(data)
    print("Iterating columns...")
    for col_label, col in data.items():
        print(col_label, col, sep="\n", end="\n\n")
    print("Iterating rows...")
    for row_label, row in data.iterrows():
        print(row_label, row, sep="\n", end="\n\n")
    #nt = namedtuple("C++", [123, 456])
    #print(nt)
    print("Iterating rows using named tuples...")
    for row in data.itertuples():
        #print(row)
        print(f"Name: {row.Name}, City: {row.City}, Age: {row.Age}, C++: {row._4}, C#: {row._6}, Python: {row.Python}")

def DataFrameTimeSeries():
    print(f"\n=== {DataFrameTimeSeries.__name__} ===")
    temp = [8.0,7.1,6.8,6.4,6.0,5.4,4.8,5.0,
                    9.1,12.8,15.3,19.1,21.2,22.1,22.4,23.1,
                    21.0,17.9,15.5,14.4,11.9,11.3,20.2,9.1]
    dateRange = pd.date_range(start='2025--2-13 00:00:00', periods=24, freq='h')
    temperatures = pd.DataFrame(data={'Temp (C)': temp}, index=dateRange)
    print(temperatures)
    print("\nTemperatures from 09:00 to 18:00") #Slicing
    print(temperatures['2025-02-13 09':'2025-02-13 18'])
    print("\nMean temperatures every 6h:")
    print(temperatures.resample(rule='6h').mean())
    print("\nTemperatures smoothed out every 3h:")
    print(temperatures.rolling(window=3).mean())
    print("\nTemperatures smoothed out every 3h (Center):")
    print(temperatures.rolling(window=3, center=True).mean())

def DataFramePlotting():
    """
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.line.html
    https://matplotlib.org/stable/api/pyplot_summary.html
    """
    print(f"\n=== {DataFramePlotting.__name__} ===")
    temp = [8.0,7.1,6.8,6.4,6.0,5.4,4.8,5.0,
                    9.1,12.8,15.3,19.1,21.2,22.1,22.4,23.1,
                    21.0,17.9,15.5,14.4,11.9,11.3,20.2,9.1]
    dateRange = pd.date_range(start='2025--2-13 00:00:00', periods=24, freq='h')
    temperatures = pd.DataFrame(data={'Temp (C)': temp}, index=dateRange)
    print(temperatures)
    temperatures.plot.line(color='b', marker='.', markerfacecolor='r', markersize=10).get_figure().savefig('/tmp/temperatures.png')
    data = {
        'Name': ['Xavier', 'Ann', 'Jana', 'Yi', 'Robin', 'Amal', 'Nori'],
        'City': ['Mexico City', 'Toronto', 'Prague', 'Shanghai', 'Manchester', 'Cairo', 'Osaka'],
        'Age': [41, 28, 33, 35, 37, 38, 45],
        'Python': [88.5, 79.0, 81.5, 80.0, 68.5, 61.0, 84.5]
    }
    indices = range(101,108)
    data = pd.DataFrame(data, index = indices)
    print(f"data ({id(data)}), ndim: {data.ndim}, size: {data.size}, shape: {data.shape}")
    data["C++"] = [78.5, 69.0, 85.5, 83.0, 72.5, 68.0, 85.5]
    data.insert(loc=3, column="C#",value= [75.0, 83.5, 69.5, 81.5, 73.5, 68.5, 84.5])
    print(data)
    data[['C++', 'C#', 'Python']].plot.hist(bins=3, alpha=0.5).get_figure().savefig('/tmp/job_candidates.png')

def DataFrameGroupBy():
    print(f"\n=== {DataFrameGroupBy.__name__} ===")
    dtypes = {
        "first_name": "category", #Panda category type which looks for commonality
        "gender": "category",
        "type": "category",
        "state": "category",
        "party": "category"
    }
    data = pd.read_csv("data/legislators-historical.csv", dtype=dtypes,usecols=list(dtypes)+["birthday","last_name"],parse_dates=["birthday"])
    print(f"data ({id(data)}), ndim: {data.ndim}, size: {data.size}, shape: {data.shape}, types: {data.dtypes}")
    print(data.head())
    state_count = data.groupby("state",observed=True).count()
    print("\nstate legislators count:")
    print(state_count)
    state_gender_count = data.groupby(["state", "gender"],observed=True).count() # This will return a Series object using MultiIndex
    print("\nstate/gender legislators count:")
    print(state_gender_count)
    print("\nstate_gender_count index:")
    print(state_gender_count.index[:5])
#####
numpy_dataframe()
DataFrameAttributes()
DataFrameAccess()
DataFrameModifications()
DataFrameArithmetic()
DataFrameFiltering()
DataFrameStatistics()
DataFrameMissingDataHandling()
DataFrameIteration()
DataFrameTimeSeries()
DataFramePlotting()
DataFrameGroupBy()