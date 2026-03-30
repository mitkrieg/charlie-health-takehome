import pandas as pd

df = pd.read_csv("./data/data_clean.csv")

df[["city", "city_lat", "city_lon"]].drop_duplicates().sort_values("city").to_csv(
    "./data/geocoding_cache.csv", index=False
)
