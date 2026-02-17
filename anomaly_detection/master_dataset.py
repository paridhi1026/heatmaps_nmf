import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split


# -----------------------------
# 1️⃣ Build badmap (list of lists)
# -----------------------------
def build_badmap(excel_path):
    df_excel = pd.read_excel(excel_path)

    badmap = {}

    for _, row in df_excel.iterrows():
        run = int(row["Run"])
        start_ls = int(row["First LS"])
        end_ls = int(row["Last LS"])

        if run not in badmap:
            badmap[run] = []

        # append as list (not tuple)
        badmap[run].append([start_ls, end_ls])

    return badmap


# -----------------------------
# 2️⃣ Label parquet dataset
# -----------------------------
def label_dataset(parquet_path, badmap):
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    df["is_anomaly"] = 0

    for run, ls_ranges in badmap.items():
        for start_ls, end_ls in ls_ranges:
            mask = (
                (df["run_number"] == run) &
                (df["ls_number"] >= start_ls) &
                (df["ls_number"] <= end_ls)
            )
            df.loc[mask, "is_anomaly"] = 1

    return df


# -----------------------------
# 3️⃣ Split dataset (by run)
# -----------------------------
def split_dataset(df):
    runs = df["run_number"].unique()

    train_runs, test_runs = train_test_split(
        runs,
        test_size=0.4,
        random_state=42
    )

    train_df = df[df["run_number"].isin(train_runs)]
    test_df  = df[df["run_number"].isin(test_runs)]

    return train_df, test_df


# -----------------------------
# 4️⃣ Main execution
# -----------------------------
if __name__ == "__main__":

    parquet_path = "ZeroBias-Run2025C-PromptReco-v1-DQMIO-Tracking-TrackParameters-highPurityTracks-pt_1-GeneralProperties-TrackEtaPhi_ImpactPoint_GenTk.parquet"
    excel_path   = "Pixel_anomalies_4Sep25.xlsx"

    badmap = build_badmap(excel_path)

    print("sample badmap entry:")
    example_run = list(badmap.keys())[0]
    print(example_run, "->", badmap[example_run])

    df_master = label_dataset(parquet_path, badmap)

    print("Class distribution:")
    print(df_master["is_anomaly"].value_counts())

    train_df, test_df = split_dataset(df_master)

    df_master.to_parquet("master_dataset.parquet", index=False)
    train_df.to_parquet("train_dataset.parquet", index=False)
    test_df.to_parquet("test_dataset.parquet", index=False)

    print("Master, train, and test datasets saved.")
