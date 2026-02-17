import pyarrow.parquet as pq

table = pq.read_table("test_predictions.parquet")
df = table.to_pandas()

print(df.head())
print(df.columns)
print(df.shape)

print(df[df["is_anomaly"]==1])
print(df.groupby("run_number")["is_anomaly"].sum())


