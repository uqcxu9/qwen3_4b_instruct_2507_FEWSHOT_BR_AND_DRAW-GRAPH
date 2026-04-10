import os
import numpy as np
import matplotlib.pyplot as plt

BASE_PATH = "/workspace/QWEN2.5_42_7b_BUFFER/draw_graph"
os.makedirs(BASE_PATH, exist_ok=True)

REAL_COLOR = "black"
MODEL1_COLOR = "#1b9e77"
MODEL2_COLOR = "#d95f02"
MODEL3_COLOR = "#7570b3"
MODEL4_COLOR = "#e7298a"
MODEL5_COLOR = "#66a61e"
MODEL6_COLOR = "#a6761d"
MODEL7_COLOR = "#666666"
MODEL8_COLOR = "#1f78b4"
MODEL9_COLOR = "#b15928"
MODEL10_COLOR = "#6a3d9a"
MODEL11_COLOR = "#ff7f00"
MODEL12_COLOR = "#33a02c"
MODEL13_COLOR = "#cab2d6"
MODEL14_COLOR = "#fb9a99"
MODEL15_COLOR = "#b2df8a"
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
# Source sequence provided by user was 2000-2020; keep 2002-2020 only
# =========================
real_us_price_inflation = [
    1.586031627, 2.270094973, 2.677236693, 3.392746845, 3.225944101,
    2.852672482, 3.839100297, -0.355546266, 1.640043442, 3.156841569,
    2.069337265, 1.464832656, 1.622222977, 0.118627136, 1.261583206,
    2.130110004, 2.442583297, 1.812210075, 1.233584396
]

# =========================
# BASE1-BASE5 price inflation (%), 2002-2020
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

qwen6_price_inflation = [
    13.17, 14.32, 3.69, 1.43, 2.51, 0.21, -1.40, 1.00, 0.48,
    0.79, 0.22, 1.03, 1.01, 1.99, 0.81, -2.13, -0.12, -1.42, -0.89
]

qwen7_price_inflation = [
    5.89, 7.92, -1.95, 1.96, 3.08, 0.03, 1.84, 4.52, -1.43,
    -6.23, -3.29, 1.91, -0.68, 1.73, 2.18, -0.37, 0.94, 2.10, -1.79
]

qwen8_price_inflation = [
    3.48, 9.08, 0.80, 1.27, 0.03, 1.41, 0.83, -0.93, -3.09,
    -0.62, -0.35, 1.87, 1.47, -0.00, 0.34, 0.80, 2.29, -0.62, -0.10
]

qwen9_price_inflation = [
    8.46, 9.00, 3.56, 3.92, 3.56, 2.38, 3.08, 2.09, 0.71,
    1.38, 1.90, 1.10, -2.91, 1.50, 2.08, -0.75, -2.53, -1.05, 0.00
]

qwen10_price_inflation = [
    7.67, 9.68, 5.65, 3.53, 7.36, 2.92, -0.14, 2.51, 1.77,
    0.70, 1.52, -0.54, 0.28, 1.70, 1.54, 0.97, -0.75, 0.10, 0.52
]

qwen11_price_inflation = [
    11.03, 8.96, 2.67, 1.64, 3.10, 5.85, 0.35, 0.91, 2.84,
    0.00, 1.62, 2.53, 0.29, 1.57, 0.94, 1.02, 1.16, 0.20, 0.57
]

qwen12_price_inflation = [
    6.05, 15.67, 12.02, 7.09, 3.77, 0.98, 1.82, 0.30, -1.81,
    -1.66, -0.90, -0.65, 0.49, 0.24, -3.54, -2.65, 2.93, 2.36, 1.21
]

qwen13_price_inflation = [
    9.94, 11.66, 10.83, 2.33, -0.15, 2.49, 1.04, 0.36, 4.06,
    2.18, -1.10, -0.35, -0.08, -4.00, 0.34, 0.40, -2.75, 2.01, 3.08
]

qwen14_price_inflation = [
    7.40, 9.49, 2.68, 1.63, 3.20, 0.18, 0.96, 3.70, 1.98,
    -0.46, 0.55, 1.26, 0.72, 0.40, 0.11, 1.05, -1.35, -0.60, 1.35
]
qwen15_price_inflation = [
    11.29, 9.74, 3.22, 1.85, 1.33, -0.11, -1.46, 1.05, 0.89,
    0.41, 2.79, 3.71, 1.08, 1.72, 2.86, 1.48, 0.44, 0.44, 1.07
]
# =========================
# Safety checks
# =========================
series_map = {
    "Real US": real_us_price_inflation,
    "BASE1": qwen1_price_inflation,
    "BASE2": qwen2_price_inflation,
    "BASE3": qwen3_price_inflation,
    "BASE4": qwen4_price_inflation,
    "BASE5": qwen5_price_inflation,
    "QWEN2.5-7B_FEWSHOT_INCOME_WEALTH_1": qwen6_price_inflation,
    "QWEN2.5-7B_FEWSHOT_INCOME_WEALTH_2": qwen7_price_inflation,
    "QWEN2.5-7B_FEWSHOT_INCOME_WEALTH_3": qwen8_price_inflation,
    "QWEN2.5-7B_FEWSHOT_BUFFER_RATIO1": qwen9_price_inflation,
    "QWEN2.5-7B_FEWSHOT_BUFFER_RATIO2": qwen10_price_inflation,
    "QWEN2.5-7B_FEWSHOT_BUFFER_RATIO3": qwen11_price_inflation,
    "QWEN2.5-7B_FEWSHOT_INCOME_WEALTH_4": qwen12_price_inflation,
    "QWEN2.5-7B_FEWSHOT_INCOME_WEALTH_5": qwen13_price_inflation,
    "QWEN2.5-7B_FEWSHOT_BUFFER_RATIO4": qwen14_price_inflation,
    "QWEN2.5-7B_FEWSHOT_BUFFER_RATIO5": qwen15_price_inflation,
}

for name, series in series_map.items():
    if len(series) != len(years):
        raise ValueError(f"{name} length mismatch: got {len(series)}, expected {len(years)}")
base_price_inflation_mean = np.mean([
    qwen1_price_inflation,
    qwen2_price_inflation,
    qwen3_price_inflation,
    qwen4_price_inflation,
    qwen5_price_inflation
], axis=0)

income_wealth_price_inflation_mean = np.mean([
    qwen6_price_inflation,
    qwen7_price_inflation,
    qwen8_price_inflation,
    qwen12_price_inflation,
    qwen13_price_inflation
], axis=0)

buffer_ratio_price_inflation_mean = np.mean([
    qwen9_price_inflation,
    qwen10_price_inflation,
    qwen11_price_inflation,
    qwen14_price_inflation,
    qwen15_price_inflation
], axis=0)
# =========================
# Figure 2: Inflation Rate (relative year axis)
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

# ax.plot(
#     years_rel,
#     qwen1_price_inflation,
#     color=MODEL1_COLOR,
#     marker='s',
#     linewidth=2,
#     markersize=5,
#     label='QWEN2.5-7B_BASE1'
# )

# ax.plot(
#     years_rel,
#     qwen2_price_inflation,
#     color=MODEL2_COLOR,
#     marker='^',
#     linewidth=2,
#     markersize=5,
#     label='QWEN2.5-7B_BASE2'
# )

# ax.plot(
#     years_rel,
#     qwen3_price_inflation,
#     color=MODEL3_COLOR,
#     marker='v',
#     linewidth=2,
#     markersize=5,
#     label='QWEN2.5-7B_BASE3'
# )

# ax.plot(
#     years_rel,
#     qwen4_price_inflation,
#     color=MODEL4_COLOR,
#     marker='P',
#     linewidth=2,
#     markersize=5,
#     label='QWEN2.5-7B_BASE4'
# )

# ax.plot(
#     years_rel,
#     qwen5_price_inflation,
#     color=MODEL5_COLOR,
#     marker='X',
#     linewidth=2,
#     markersize=5,
#     label='QWEN2.5-7B_BASE5'
# )

# ax.plot(
#     years_rel,
#     qwen6_price_inflation,
#     color=MODEL6_COLOR,
#     marker='*',
#     linewidth=2,
#     markersize=7,
#     label='QWEN2.5-7B_FEWSHOT_INCOME_WEALTH_1'
# )

# ax.plot(
#     years_rel,
#     qwen7_price_inflation,
#     color=MODEL7_COLOR,
#     marker='h',
#     linewidth=2,
#     markersize=5,
#     label='QWEN2.5-7B_FEWSHOT_INCOME_WEALTH_2'
# )

# ax.plot(
#     years_rel,
#     qwen8_price_inflation,
#     color=MODEL8_COLOR,
#     marker='o',
#     linewidth=2,
#     markersize=5,
#     label='QWEN2.5-7B_FEWSHOT_INCOME_WEALTH_3'
# )

# ax.plot(
#     years_rel,
#     qwen9_price_inflation,
#     color=MODEL9_COLOR,
#     marker='<',
#     linewidth=2,
#     markersize=5,
#     label='QWEN2.5-7B_FEWSHOT_BUFFER_RATIO1'
# )

# ax.plot(
#     years_rel,
#     qwen10_price_inflation,
#     color=MODEL10_COLOR,
#     marker='>',
#     linewidth=2,
#     markersize=5,
#     label='QWEN2.5-7B_FEWSHOT_BUFFER_RATIO2'
# )

# ax.plot(
#     years_rel,
#     qwen11_price_inflation,
#     color=MODEL11_COLOR,
#     marker='d',
#     linewidth=2,
#     markersize=5,
#     label='QWEN2.5-7B_FEWSHOT_BUFFER_RATIO3'
# )

# ax.plot(
#     years_rel,
#     qwen12_price_inflation,
#     color=MODEL12_COLOR,
#     marker='p',
#     linewidth=2,
#     markersize=5,
#     label='QWEN2.5-7B_FEWSHOT_INCOME_WEALTH_4'
# )

# ax.plot(
#     years_rel,
#     qwen13_price_inflation,
#     color=MODEL13_COLOR,
#     marker='H',
#     linewidth=2,
#     markersize=5,
#     label='QWEN2.5-7B_FEWSHOT_INCOME_WEALTH_5'
# )

# ax.plot(
#     years_rel,
#     qwen14_price_inflation,
#     color=MODEL14_COLOR,
#     marker='8',
#     linewidth=2,
#     markersize=5,
#     label='QWEN2.5-7B_FEWSHOT_BUFFER_RATIO4'
# )
# ax.plot(
#     years_rel,
#     qwen15_price_inflation,
#     color=MODEL15_COLOR,
#     marker='1',
#     linewidth=2,
#     markersize=7,
#     label='QWEN2.5-7B_FEWSHOT_BUFFER_RATIO5'
# )
ax.plot(
    years_rel,
    base_price_inflation_mean,
    color=MODEL1_COLOR,
    marker='s',
    linewidth=2,
    markersize=5,
    label='QWEN2.5-7B_BASE Mean'
)

ax.plot(
    years_rel,
    income_wealth_price_inflation_mean,
    color=MODEL8_COLOR,
    marker='*',
    linewidth=2,
    markersize=7,
    label='QWEN2.5-7B_FEWSHOT_INCOME_WEALTH Mean'
)

ax.plot(
    years_rel,
    buffer_ratio_price_inflation_mean,
    color=MODEL11_COLOR,
    marker='>',
    linewidth=2,
    markersize=6,
    label='QWEN2.5-7B_FEWSHOT_BUFFER_RATIO Mean'
)

annotate_min_max(ax, years_rel, real_us_price_inflation, REAL_COLOR)
annotate_min_max(ax, years_rel, base_price_inflation_mean, MODEL1_COLOR)
annotate_min_max(ax, years_rel, income_wealth_price_inflation_mean, MODEL8_COLOR)
annotate_min_max(ax, years_rel, buffer_ratio_price_inflation_mean, MODEL11_COLOR)
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
plt.show()

# =========================
# Optional: same figure with calendar year axis
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

# ax.plot(
#     years,
#     qwen1_price_inflation,
#     color=MODEL1_COLOR,
#     marker='s',
#     linewidth=2,
#     markersize=5,
#     label='QWEN2.5-7B_BASE1'
# )

# ax.plot(
#     years,
#     qwen2_price_inflation,
#     color=MODEL2_COLOR,
#     marker='^',
#     linewidth=2,
#     markersize=5,
#     label='QWEN2.5-7B_BASE2'
# )

# ax.plot(
#     years,
#     qwen3_price_inflation,
#     color=MODEL3_COLOR,
#     marker='v',
#     linewidth=2,
#     markersize=5,
#     label='QWEN2.5-7B_BASE3'
# )

# ax.plot(
#     years,
#     qwen4_price_inflation,
#     color=MODEL4_COLOR,
#     marker='P',
#     linewidth=2,
#     markersize=5,
#     label='QWEN2.5-7B_BASE4'
# )

# ax.plot(
#     years,
#     qwen5_price_inflation,
#     color=MODEL5_COLOR,
#     marker='X',
#     linewidth=2,
#     markersize=5,
#     label='QWEN2.5-7B_BASE5'
# )

# ax.plot(
#     years,
#     qwen6_price_inflation,
#     color=MODEL6_COLOR,
#     marker='*',
#     linewidth=2,
#     markersize=7,
#     label='QWEN2.5-7B_FEWSHOT_INCOME_WEALTH_1'
# )

# ax.plot(
#     years,
#     qwen7_price_inflation,
#     color=MODEL7_COLOR,
#     marker='h',
#     linewidth=2,
#     markersize=5,
#     label='QWEN2.5-7B_FEWSHOT_INCOME_WEALTH_2'
# )

# ax.plot(
#     years,
#     qwen8_price_inflation,
#     color=MODEL8_COLOR,
#     marker='o',
#     linewidth=2,
#     markersize=5,
#     label='QWEN2.5-7B_FEWSHOT_INCOME_WEALTH_3'
# )

# ax.plot(
#     years,
#     qwen9_price_inflation,
#     color=MODEL9_COLOR,
#     marker='<',
#     linewidth=2,
#     markersize=5,
#     label='QWEN2.5-7B_FEWSHOT_BUFFER_RATIO1'
# )

# ax.plot(
#     years,
#     qwen10_price_inflation,
#     color=MODEL10_COLOR,
#     marker='>',
#     linewidth=2,
#     markersize=5,
#     label='QWEN2.5-7B_FEWSHOT_BUFFER_RATIO2'
# )

# ax.plot(
#     years,
#     qwen11_price_inflation,
#     color=MODEL11_COLOR,
#     marker='d',
#     linewidth=2,
#     markersize=5,
#     label='QWEN2.5-7B_FEWSHOT_BUFFER_RATIO3'
# )

# ax.plot(
#     years,
#     qwen12_price_inflation,
#     color=MODEL12_COLOR,
#     marker='p',
#     linewidth=2,
#     markersize=5,
#     label='QWEN2.5-7B_FEWSHOT_INCOME_WEALTH_4'
# )

# ax.plot(
#     years,
#     qwen13_price_inflation,
#     color=MODEL13_COLOR,
#     marker='H',
#     linewidth=2,
#     markersize=5,
#     label='QWEN2.5-7B_FEWSHOT_INCOME_WEALTH_5'
# )

# ax.plot(
#     years,
#     qwen14_price_inflation,
#     color=MODEL14_COLOR,
#     marker='8',
#     linewidth=2,
#     markersize=5,
#     label='QWEN2.5-7B_FEWSHOT_BUFFER_RATIO4'
# )
# ax.plot(
#     years,
#     qwen15_price_inflation,
#     color=MODEL15_COLOR,
#     marker='1',
#     linewidth=2,
#     markersize=7,
#     label='QWEN2.5-7B_FEWSHOT_BUFFER_RATIO5'
# )

ax.plot(
    years,
    base_price_inflation_mean,
    color=MODEL1_COLOR,
    marker='s',
    linewidth=2,
    markersize=5,
    label='QWEN2.5-7B_BASE Mean'
)

ax.plot(
    years,
    income_wealth_price_inflation_mean,
    color=MODEL8_COLOR,
    marker='*',
    linewidth=2,
    markersize=7,
    label='QWEN2.5-7B_FEWSHOT_INCOME_WEALTH Mean'
)

ax.plot(
    years,
    buffer_ratio_price_inflation_mean,
    color=MODEL11_COLOR,
    marker='>',
    linewidth=2,
    markersize=6,
    label='QWEN2.5-7B_FEWSHOT_BUFFER_RATIO Mean'
)

annotate_min_max(ax, years, real_us_price_inflation, REAL_COLOR)
annotate_min_max(ax, years, base_price_inflation_mean, MODEL1_COLOR)
annotate_min_max(ax, years, income_wealth_price_inflation_mean, MODEL8_COLOR)
annotate_min_max(ax, years, buffer_ratio_price_inflation_mean, MODEL11_COLOR)

ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
ax.set_xlabel("Calendar Year", fontsize=12)
ax.set_ylabel("Inflation Rate (%)", fontsize=12)
ax.set_title("Figure 2: Inflation Rate (2002–2020)", fontsize=14)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(BASE_PATH, "figure2_inflation_rate_all_calendar_year.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.show()