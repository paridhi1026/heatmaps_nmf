import pyarrow.parquet as pq
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

def save_ls_plots(df, run_number, outdir="plots"):
    """
    Save one PNG heatmap per LS for a given run_number.
    """
    rows = df[df["run_number"] == run_number].sort_values("ls_number")
    if rows.empty:
        print(f"No entries for run {run_number}")
        return

    os.makedirs(outdir, exist_ok=True)

    for _, row in rows.iterrows():
        # Stack 2D data
        z = np.stack(row["data"], axis=0).astype(float)
        x_edges = np.linspace(row["x_min"], row["x_max"], z.shape[0] + 1)
        y_edges = np.linspace(row["y_min"], row["y_max"], z.shape[1] + 1)

        # Plot
        fig, ax = plt.subplots(figsize=(6, 5))
        pcm = ax.pcolormesh(x_edges, y_edges, z.T, shading="auto")
        fig.colorbar(pcm, ax=ax, label="Entries")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Run {row['run_number']}, LS {row['ls_number']}")

        # Save to PNG
        fname = os.path.join(outdir, f"run{row['run_number']}_ls{row['ls_number']}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)  # free memory

        print(f"Saved {fname}")
  
# Load parquet into pandas DataFrame
table = pq.read_table(sys.argv[1])
df = table.to_pandas()
df["is_anomaly"] = [False] * len(df)
unique_runs = sorted(df["run_number"].unique())
print(unique_runs)
# Pick one run_number
run_number = 383648
save_ls_plots(df, run_number, outdir=f"plots_{run_number}")

