import numpy,polars,timeit, pandas as pd
from pathlib import Path
from utils.FileUtil import Download
# https://realpython.com/polars-python/
size = 5000
rng = numpy.random.default_rng(seed=19)
buildings_data = {
    "sqft": rng.exponential(scale=1000, size=size),
    "price": rng.exponential(scale=100_000, size=size),
    "year": rng.integers(low=1995, high=2024, size=size),
    "type": rng.choice(["A","B","C"], size=size)
}

def SelectContext():
    print(f"=== {SelectContext.__name__} ===")
    buildings = polars.DataFrame(buildings_data)
    print(f"buildings: {buildings}")
    print("sqft:")
    print(buildings.select("sqft"))
    print("sqft with expression:")
    print(buildings.select(polars.col("sqft")))
    print("sqft with expression sorted:")
    print(buildings.select(polars.col("sqft").sort()))

def FilterContext():
    print(f"\n=== {FilterContext.__name__} ===")
    buildings = polars.DataFrame(buildings_data)
    print(f"buildings: {buildings}")
    after_2015 = buildings.filter(polars.col("year") > 2015)
    print(f"Buildings after 2015: {after_2015.shape}")
    print("Min year after 2015:")
    min_after_2015 = after_2015.select(polars.col("year").min())
    print(min_after_2015)

def Aggregation():
    print(f"\n=== {Aggregation.__name__} ===")
    buildings = polars.DataFrame(buildings_data)
    print(f"buildings: {buildings}")
    building_types = buildings.group_by("type").agg(
        [
            polars.mean("sqft").alias("mean_sqft"),
            polars.median("year").alias("median_year"),
            polars.median("price").alias("median_price"),
            polars.len().alias("count")
        ]
    )
    print("Average sqft, median building year and price, and number of buildings for each building type:")
    print(building_types)

def LazyAPI():
    print(f"\n=== {LazyAPI.__name__} ===")
    buildings = polars.LazyFrame(buildings_data)
    print(f"buildings: {buildings}")
    query = (buildings.with_columns(
        (polars.col("price") / polars.col("sqft")).alias("price_per_sqft")
    ).filter(polars.col("price_per_sqft") > 100)
    .filter(polars.col("year") < 2020)
    )
    print(f"Lazy query:")
    print(query.explain())
    #print("Lazy query plan:")
    #print(query.show_graph())
    print("Query result:")
    result = query.collect()
    print(result)
    print("Query result summary:")
    print(result.describe())

def ScanLargeData(url, path):
    print(f"\n=== {ScanLargeData.__name__} ===")
    Download(url, Path(path))
    data = polars.scan_csv(Path(path))
    print(f"data: {data}")
    query = (
        data
        .filter((polars.col("Model Year") >= 2018))
        .filter(
            polars.col("Electric Vehicle Type") == "Battery Electric Vehicle (BEV)"
        )
        .group_by(["State", "Make"])
        .agg(
            polars.mean("Electric Range").alias("Average Electric Range"),
            polars.min("Model Year").alias("Oldest Model Year"),
            polars.len().alias("Number of Cars"),
        )
        .filter(polars.col("Average Electric Range") > 0)
        .filter(polars.col("Number of Cars") > 5)
        .sort(polars.col("Number of Cars"), descending=True)
    )
    print(f"Lazy query:")
    print(query.explain())
    #print("Lazy query plan:")
    #print(query.show_graph())
    result = query.collect()
    print("Query result summary:")
    print(result.describe())
    print("Query result:")
    print(result)

def ScanLargeDataPandas(url, path):
    print(f"\n=== {ScanLargeDataPandas.__name__} ===")
    Download(url, Path(path))
    data = pd.read_csv(path)
    print(f"data ({id(data)}), ndim: {data.ndim}, size: {data.size}, shape: {data.shape}")
    print("\ndata.describe():")
    print(data.describe())
    print("\ndata.info():")
    data.info()
    print(data.dtypes)
    print(data.head())
    filter = (data["Model Year"] >= 2018) & (data["Electric Vehicle Type"] == "Battery Electric Vehicle (BEV)")
    data = data[filter].groupby(["State", "Make"], sort=False, observed=True, as_index=False).agg( avg_electric_range=pd.NamedAgg(column="Electric Range", aggfunc="mean"), oldest_model_year=pd.NamedAgg(column="Model Year", aggfunc="min"), count=pd.NamedAgg(column="Model Year", aggfunc="count"))
    filter = (data["avg_electric_range"] > 0) & (data["count"] > 5)
    data = data[filter].sort_values(by=["count"], ascending=[False]) # Tie-breaks in C++ scores between Jana and Nori
    print(data.head(5))
    print(data.tail(5))

def PolarsLargeData(path):
    data = polars.scan_csv(Path(path))
    query = (
        data
        .filter((polars.col("Model Year") >= 2018))
        .filter(
            polars.col("Electric Vehicle Type") == "Battery Electric Vehicle (BEV)"
        )
        .group_by(["State", "Make"])
        .agg(
            polars.mean("Electric Range").alias("Average Electric Range"),
            polars.min("Model Year").alias("Oldest Model Year"),
            polars.len().alias("Number of Cars"),
        )
        .filter(polars.col("Average Electric Range") > 0)
        .filter(polars.col("Number of Cars") > 5)
        .sort(polars.col("Number of Cars"), descending=True)
    )
    query.collect()

def PandasLargeData(path):
    data = pd.read_csv(path)
    filter = (data["Model Year"] >= 2018) & (data["Electric Vehicle Type"] == "Battery Electric Vehicle (BEV)")
    data = data[filter].groupby(["State", "Make"], sort=False, observed=True, as_index=False).agg( avg_electric_range=pd.NamedAgg(column="Electric Range", aggfunc="mean"), oldest_model_year=pd.NamedAgg(column="Model Year", aggfunc="min"), count=pd.NamedAgg(column="Model Year", aggfunc="count"))
    filter = (data["avg_electric_range"] > 0) & (data["count"] > 5)
    data = data[filter].sort_values(by=["count"], ascending=[False]) # Tie-breaks in C++ scores between Jana and Nori

def PandasPolarBenchmark(url, path):
    print(f"\nPerformance comparison between Pandas and Polars:")
    Download(url, Path(path))
    t1 = timeit.Timer(lambda: PandasLargeData(path))
    t2 = timeit.Timer(lambda: PolarsLargeData(path))
    print(f"Pandas: {t1.timeit(number=10)}s, Polars: {t2.timeit(number=10)}s")

def PrepareDataForChromaDB(path, vehicle_years: list[int] = [2017]):
    """
    Prepare the car reviews dataset for ChromaDB
    https://realpython.com/chromadb-vector-database/
    https://www.kaggle.com/datasets/ankkur13/edmundsconsumer-car-ratings-and-reviews?resource=download
    """
    print(f"\n=== {PrepareDataForChromaDB.__name__} ===")
    # Define the schema to ensure proper data types are enforced
    dtypes = {
        "": polars.Int64,
        "Review_Date": polars.Utf8,
        "Author_Name": polars.Utf8,
        "Vehicle_Title": polars.Utf8,
        "Review_Title": polars.Utf8,
        "Review": polars.Utf8,
        "Rating": polars.Float64,
    }

    # Scan the car reviews dataset(s)
    car_reviews = polars.scan_csv(Path(path), schema_overrides = dtypes)

    # Extract the vehicle title and year as new columns
    # Filter on selected years
    car_review_db_data = (
        car_reviews.with_columns(
            [
                (
                    polars.col("Vehicle_Title").str.split(
                        by=" ").list.get(0).cast(polars.Int64)
                ).alias("Vehicle_Year"),
                (polars.col("Vehicle_Title").str.split(by=" ").list.get(1)).alias(
                    "Vehicle_Model"
                ),
            ]
        )
        .filter(polars.col("Vehicle_Year").is_in(vehicle_years))
        .select(["Review_Title", "Review", "Rating", "Vehicle_Year", "Vehicle_Model"])
        .sort(["Vehicle_Model", "Rating"])
        .collect()
    )

    # Create ids, documents, and metadatas data in the format chromadb expects
    ids = [f"review{i}" for i in range(car_review_db_data.shape[0])]
    documents = car_review_db_data["Review"].to_list()
    metadatas = car_review_db_data.drop("Review").to_dicts()
    return {"ids": ids, "documents": documents, "metadatas": metadatas}

def ChromaDBCarReviews():
    print(f"\n=== {ChromaDBCarReviews.__name__} ===")
    reviews = PrepareDataForChromaDB("data/Scraped_Car_Review_dodge.csv")
    print(f"keys: {reviews.keys()}")
    print(f"reviews[-10] ids: {reviews['ids'][-10]}, metadata: {reviews['metadatas'][-10]}")
    print(f"reviews[-10] documents: {reviews['documents'][-10]}")

if __name__ == "__main__":
    SelectContext()
    FilterContext()
    Aggregation()
    LazyAPI()
    ScanLargeData("https://data.wa.gov/api/views/f6w7-q2d2/rows.csv?accessType=DOWNLOAD", "/tmp/electric_cars.csv")
    ScanLargeDataPandas("https://data.wa.gov/api/views/f6w7-q2d2/rows.csv?accessType=DOWNLOAD", "/tmp/electric_cars.csv")
    PandasPolarBenchmark("https://data.wa.gov/api/views/f6w7-q2d2/rows.csv?accessType=DOWNLOAD", "/tmp/electric_cars.csv")
    ChromaDBCarReviews()