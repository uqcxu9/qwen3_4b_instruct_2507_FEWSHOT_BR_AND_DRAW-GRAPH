import os
import numpy as np
import matplotlib.pyplot as plt

BASE_PATH = "/workspace/QWEN2.5_42_7b_main/draw_graph"
os.makedirs(BASE_PATH, exist_ok=True)

REAL_COLOR = "black"
BASE7B_COLOR = "#1b9e77"
QWEN3_4B_COLOR = "#7570b3"
BR4B_COLOR = "#d95f02"

# =========================
# Year alignment
# Figure 2 inflation panel uses 2002-2020 -> 19 points
# =========================
years = list(range(2002, 2021))
years_rel = list(range(1, 20))  # relative years for 2002-2020


def annotate_min_max(ax, x, y, color, dy_max=6, dy_min=-10):
    y = np.array(y)
    idx_max = int(np.argmax(y))
    idx_min = int(np.argmin(y))

    ax.annotate(
        f"Max: {y[idx_max]:.2f}",
        xy=(x[idx_max], y[idx_max]),
        xytext=(6, dy_max),
        textcoords="offset points",
        color=color,
        fontsize=9
    )

    ax.annotate(
        f"Min: {y[idx_min]:.2f}",
        xy=(x[idx_min], y[idx_min]),
        xytext=(6, dy_min),
        textcoords="offset points",
        color=color,
        fontsize=9
    )

# =========================
# Real US price inflation (%), 2002-2020
# =========================
real_us_price_inflation = [
    1.586031627, 2.270094973, 2.677236693, 3.392746845, 3.225944101,
    2.852672482, 3.839100297, -0.355546266, 1.640043442, 3.156841569,
    2.069337265, 1.464832656, 1.622222977, 0.118627136, 1.261583206,
    2.130110004, 2.442583297, 1.812210075, 1.233584396
]

# =========================
# 7B Base (qwen1-qwen5) price inflation (%), 2002-2020
# =========================
qwen1_price_inflation = [
    2.94, 17.44, 16.43, 10.96, 9.53, 8.70, 3.48, 1.93, -0.55,
    -16.44, -10.14, 3.87, 3.47, 2.19, 1.91, 0.53, -0.40, -1.04, -1.73
]

qwen2_price_inflation = [
    -12.20, 11.18, 10.89, 10.40, 10.10, 4.94, 0.71, -4.84, -3.02,
    -2.49, -3.86, -1.07, -4.21, -0.60, 3.15, 2.37, -1.72, -0.71, -1.50
]

qwen3_price_inflation = [
    -6.29, 15.89, 11.62, 12.57, 13.26, 5.48, 2.18, -6.38, -15.49,
    0.37, 2.72, 3.58, 1.83, -3.53, 0.14, 0.32, -5.42, 0.57, 1.97
]

qwen4_price_inflation = [
    -9.60, 10.20, 11.81, 10.80, 5.27, 2.58, -1.61, -7.77, -1.76,
    -1.84, 0.72, -0.16, -4.55, -0.59, 2.17, 1.81, 0.68, -3.05, 0.62
]

qwen5_price_inflation = [
    -10.90, 6.29, 6.21, 5.28, 4.66, 4.89, 3.27, -2.02, -0.65,
    -4.18, -0.47, 0.99, -8.74, 0.29, 2.05, 1.76, 2.31, -2.04, -0.56
]

# =========================
# Qwen3-4B-Instruct price inflation (%), 2002-2020
# =========================
q3_4b_1_price_inflation = [-2.61, 22.51, 18.55, -0.40, -12.46, -2.61, -0.50, 3.36, 7.88, 1.92, -7.01, -7.79, 1.83, 8.15, 3.87, -4.00, -5.70, 1.02, 5.05]
q3_4b_2_price_inflation = [-6.09, 15.26, 8.79, 2.23, -6.54, -8.46, 1.99, 3.43, 2.67, -1.45, -5.92, -0.35, 4.06, 0.55, -2.44, -1.38, 2.03, 2.99, 2.64]
q3_4b_3_price_inflation = [-18.15, 9.16, 21.49, 15.86, 0.99, -12.84, -3.32, 4.44, 1.50, -4.34, -2.25, 2.47, 4.21, 3.15, -0.11, -4.74, -1.29, 2.17, 2.09]
q3_4b_4_price_inflation = [-1.02, 25.50, 16.15, -1.53, -9.71, -5.92, 2.27, 3.95, -0.12, -2.70, -5.48, 3.04, 4.23, 0.20, -3.78, 2.20, 0.54, -0.38, 0.22]
q3_4b_5_price_inflation = [-1.88, 15.67, 21.37, -1.22, -17.50, -5.88, 2.91, 6.54, 0.95, -9.35, -3.55, 8.78, 5.71, -0.06, -7.49, -2.37, 4.21, 7.00, 2.56]

# =========================
# Qwen3-4B Fewshot BR price inflation (%), 2002-2020
# =========================
br4b_1_price_inflation = [-4.12, 7.36, 4.56, 1.22, -1.02, -2.13, -0.67, 0.10, 0.39, 0.66, 0.18, 0.01, -0.46, -0.41, 0.43, 0.18, -0.35, -0.21, 0.76]
br4b_2_price_inflation = [2.81, 12.20, 6.84, -2.83, -3.85, 1.79, 0.33, -0.78, 0.42, -1.48, -0.60, 0.89, 0.26, -0.67, 0.25, 0.83, -1.37, -0.19, 0.87]
br4b_3_price_inflation = [1.07, 4.78, 6.66, 7.72, 2.53, -2.71, -2.14, 1.29, 0.45, 0.13, -1.01, -0.70, -0.01, 0.48, 0.57, -0.74, -0.35, 0.63, -0.09]
br4b_4_price_inflation = [-2.18, 7.76, 7.03, 3.91, -2.59, -4.07, 1.63, 0.48, -0.18, 0.11, -0.41, -0.26, -0.44, 0.05, 0.74, -0.31, -1.01, 0.63, 0.67]
br4b_5_price_inflation = [1.53, 8.73, 6.47, 4.89, -1.25, -5.34, 2.59, 1.04, -0.97, -0.26, -0.43, -0.36, 0.11, 0.89, 0.36, 0.43, -1.32, -0.94, 0.00]

# =========================
# Safety checks
# =========================
series_map = {
    "Real US": real_us_price_inflation,
    "7B Base 1": qwen1_price_inflation,
    "7B Base 2": qwen2_price_inflation,
    "7B Base 3": qwen3_price_inflation,
    "7B Base 4": qwen4_price_inflation,
    "7B Base 5": qwen5_price_inflation,
    "Qwen3-4B Run 1": q3_4b_1_price_inflation,
    "Qwen3-4B Run 2": q3_4b_2_price_inflation,
    "Qwen3-4B Run 3": q3_4b_3_price_inflation,
    "Qwen3-4B Run 4": q3_4b_4_price_inflation,
    "Qwen3-4B Run 5": q3_4b_5_price_inflation,
    "4B Fewshot BR Run 1": br4b_1_price_inflation,
    "4B Fewshot BR Run 2": br4b_2_price_inflation,
    "4B Fewshot BR Run 3": br4b_3_price_inflation,
    "4B Fewshot BR Run 4": br4b_4_price_inflation,
    "4B Fewshot BR Run 5": br4b_5_price_inflation,
}

for name, series in series_map.items():
    if len(series) != len(years):
        raise ValueError(f"{name} length mismatch: got {len(series)}, expected {len(years)}")

# =========================
# Compute means
# =========================
base_price_inflation_mean = np.mean([
    qwen1_price_inflation,
    qwen2_price_inflation,
    qwen3_price_inflation,
    qwen4_price_inflation,
    qwen5_price_inflation
], axis=0)

q3_4b_price_inflation_mean = np.mean([
    q3_4b_1_price_inflation,
    q3_4b_2_price_inflation,
    q3_4b_3_price_inflation,
    q3_4b_4_price_inflation,
    q3_4b_5_price_inflation
], axis=0)

br4b_price_inflation_mean = np.mean([
    br4b_1_price_inflation,
    br4b_2_price_inflation,
    br4b_3_price_inflation,
    br4b_4_price_inflation,
    br4b_5_price_inflation
], axis=0)

# =========================
# Figure 1: Inflation Rate (relative year axis)
# =========================
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    years_rel,
    real_us_price_inflation,
    color=REAL_COLOR,
    marker='D',
    linewidth=2,
    markersize=5,
    label='Real US'
)

ax.plot(
    years_rel,
    base_price_inflation_mean,
    color=BASE7B_COLOR,
    marker='s',
    linewidth=2,
    markersize=5,
    label='QWEN2.5-7B Base Mean'
)

ax.plot(
    years_rel,
    q3_4b_price_inflation_mean,
    color=QWEN3_4B_COLOR,
    marker='^',
    linewidth=2,
    markersize=5,
    label='Qwen3-4B-Instruct Mean'
)

ax.plot(
    years_rel,
    br4b_price_inflation_mean,
    color=BR4B_COLOR,
    marker='X',
    linewidth=2,
    markersize=5,
    label='4B Fewshot BR Mean'
)

annotate_min_max(ax, years_rel, real_us_price_inflation, REAL_COLOR)
annotate_min_max(ax, years_rel, base_price_inflation_mean, BASE7B_COLOR)
annotate_min_max(ax, years_rel, q3_4b_price_inflation_mean, QWEN3_4B_COLOR)
annotate_min_max(ax, years_rel, br4b_price_inflation_mean, BR4B_COLOR)

ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Inflation Rate (%)", fontsize=12)
ax.set_title("Figure 2: Inflation Rate", fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(BASE_PATH, "figure2_inflation_rate_all.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# =========================
# Figure 2: Same figure with calendar year axis
# =========================
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    years,
    real_us_price_inflation,
    color=REAL_COLOR,
    marker='D',
    linewidth=2,
    markersize=5,
    label='Real US'
)

ax.plot(
    years,
    base_price_inflation_mean,
    color=BASE7B_COLOR,
    marker='s',
    linewidth=2,
    markersize=5,
    label='QWEN2.5-7B Base Mean'
)

ax.plot(
    years,
    q3_4b_price_inflation_mean,
    color=QWEN3_4B_COLOR,
    marker='^',
    linewidth=2,
    markersize=5,
    label='Qwen3-4B-Instruct Mean'
)

ax.plot(
    years,
    br4b_price_inflation_mean,
    color=BR4B_COLOR,
    marker='X',
    linewidth=2,
    markersize=5,
    label='4B Fewshot BR Mean'
)

annotate_min_max(ax, years, real_us_price_inflation, REAL_COLOR)
annotate_min_max(ax, years, base_price_inflation_mean, BASE7B_COLOR)
annotate_min_max(ax, years, q3_4b_price_inflation_mean, QWEN3_4B_COLOR)
annotate_min_max(ax, years, br4b_price_inflation_mean, BR4B_COLOR)

ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
ax.set_xlabel("Calendar Year", fontsize=12)
ax.set_ylabel("Inflation Rate (%)", fontsize=12)
ax.set_title("Figure 2: Inflation Rate (2002-2020)", fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(BASE_PATH, "figure2_inflation_rate_all_calendar_year.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# =========================
# Statistics
# =========================
print("\n=== Price Inflation Statistics ===")

# Real US
real_arr = np.array(real_us_price_inflation)
print(f"\nReal US:")
print(f"  Max:  {real_arr.max():.4f}")
print(f"  Min:  {real_arr.min():.4f}")
print(f"  Mean: {real_arr.mean():.4f}")
print(f"  Std:  {real_arr.std():.4f}")

# 7B Base per-run
base_runs = {
    "7B Base Run 1": qwen1_price_inflation,
    "7B Base Run 2": qwen2_price_inflation,
    "7B Base Run 3": qwen3_price_inflation,
    "7B Base Run 4": qwen4_price_inflation,
    "7B Base Run 5": qwen5_price_inflation,
}
print("\n--- 7B Base (per-run) ---")
for name, data in base_runs.items():
    arr = np.array(data)
    print(f"\n{name}:")
    print(f"  Max:  {arr.max():.4f}")
    print(f"  Min:  {arr.min():.4f}")
    print(f"  Mean: {arr.mean():.4f}")
    print(f"  Std:  {arr.std():.4f}")

# 7B Base overall (pooled)
base_pooled = np.concatenate([np.array(d) for d in base_runs.values()])
print(f"\n7B Base Overall (all 5 runs pooled):")
print(f"  Max:  {base_pooled.max():.4f}")
print(f"  Min:  {base_pooled.min():.4f}")
print(f"  Mean: {base_pooled.mean():.4f}")
print(f"  Std:  {base_pooled.std():.4f}")

# 7B Base mean series
base_mean_arr = np.array(base_price_inflation_mean)
print(f"\n7B Base Mean Series:")
print(f"  Max:  {base_mean_arr.max():.4f}")
print(f"  Min:  {base_mean_arr.min():.4f}")
print(f"  Mean: {base_mean_arr.mean():.4f}")
print(f"  Std:  {base_mean_arr.std():.4f}")

# Qwen3-4B per-run
q3_4b_runs = {
    "Qwen3-4B Run 1": q3_4b_1_price_inflation,
    "Qwen3-4B Run 2": q3_4b_2_price_inflation,
    "Qwen3-4B Run 3": q3_4b_3_price_inflation,
    "Qwen3-4B Run 4": q3_4b_4_price_inflation,
    "Qwen3-4B Run 5": q3_4b_5_price_inflation,
}
print("\n--- Qwen3-4B-Instruct (per-run) ---")
for name, data in q3_4b_runs.items():
    arr = np.array(data)
    print(f"\n{name}:")
    print(f"  Max:  {arr.max():.4f}")
    print(f"  Min:  {arr.min():.4f}")
    print(f"  Mean: {arr.mean():.4f}")
    print(f"  Std:  {arr.std():.4f}")

# Qwen3-4B overall (pooled)
q3_4b_pooled = np.concatenate([np.array(d) for d in q3_4b_runs.values()])
print(f"\nQwen3-4B Overall (all 5 runs pooled):")
print(f"  Max:  {q3_4b_pooled.max():.4f}")
print(f"  Min:  {q3_4b_pooled.min():.4f}")
print(f"  Mean: {q3_4b_pooled.mean():.4f}")
print(f"  Std:  {q3_4b_pooled.std():.4f}")

# Qwen3-4B mean series
q3_4b_mean_arr = np.array(q3_4b_price_inflation_mean)
print(f"\nQwen3-4B Mean Series:")
print(f"  Max:  {q3_4b_mean_arr.max():.4f}")
print(f"  Min:  {q3_4b_mean_arr.min():.4f}")
print(f"  Mean: {q3_4b_mean_arr.mean():.4f}")
print(f"  Std:  {q3_4b_mean_arr.std():.4f}")

# 4B Fewshot BR per-run
br4b_runs = {
    "4B Fewshot BR Run 1": br4b_1_price_inflation,
    "4B Fewshot BR Run 2": br4b_2_price_inflation,
    "4B Fewshot BR Run 3": br4b_3_price_inflation,
    "4B Fewshot BR Run 4": br4b_4_price_inflation,
    "4B Fewshot BR Run 5": br4b_5_price_inflation,
}
print("\n--- 4B Fewshot BR (per-run) ---")
for name, data in br4b_runs.items():
    arr = np.array(data)
    print(f"\n{name}:")
    print(f"  Max:  {arr.max():.4f}")
    print(f"  Min:  {arr.min():.4f}")
    print(f"  Mean: {arr.mean():.4f}")
    print(f"  Std:  {arr.std():.4f}")

# 4B Fewshot BR overall (pooled)
br4b_pooled = np.concatenate([np.array(d) for d in br4b_runs.values()])
print(f"\n4B Fewshot BR Overall (all 5 runs pooled):")
print(f"  Max:  {br4b_pooled.max():.4f}")
print(f"  Min:  {br4b_pooled.min():.4f}")
print(f"  Mean: {br4b_pooled.mean():.4f}")
print(f"  Std:  {br4b_pooled.std():.4f}")

# 4B Fewshot BR mean series
br4b_mean_arr = np.array(br4b_price_inflation_mean)
print(f"\n4B Fewshot BR Mean Series:")
print(f"  Max:  {br4b_mean_arr.max():.4f}")
print(f"  Min:  {br4b_mean_arr.min():.4f}")
print(f"  Mean: {br4b_mean_arr.mean():.4f}")
print(f"  Std:  {br4b_mean_arr.std():.4f}")
