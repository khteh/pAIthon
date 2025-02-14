import numpy,polars,pathlib,magic
from pathlib import Path
from downloads import download_file
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
    print(f"\n=== {SelectContext.__name__} ===")
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

def ScanLargeData(url, local):
    print(f"\n=== {ScanLargeData.__name__} ===")
    if not Path(local).exists() or not Path(local).is_file():
        download_file(url, Path(local))
    print(f"file magic: {magic.from_file(local)}")
    data = polars.scan_csv(Path(local))
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
         polars.count().alias("Number of Cars"),
     )
     .filter(polars.col("Average Electric Range") > 0)
     .filter(polars.col("Number of Cars") > 5)
     .sort(polars.col("Number of Cars"), descending=True)
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
   
###
SelectContext()
FilterContext()
Aggregation()
LazyAPI()
ScanLargeData("https://data.wa.gov/api/views/f6w7-q2d2/rows.csv?accessType=DOWNLOAD", "/tmp/electric_cars.csv")