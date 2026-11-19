import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import matplotlib.pyplot as plt
import aerosol.plotting as ap
import aerosol.lab as al

import matplotlib
import matplotlib.cm as cm

if not hasattr(cm, "get_cmap"):
    cm.get_cmap = matplotlib.colormaps.get_cmap

ds = xr.open_dataset("/home/kivekia/Python/beijing_all.nc")

ds = ds.sortby('time')

negp_og=ds["neg_particles"]
negi_og=ds["neg_ions"]

negp_og=negp_og.copy()
negi_og=negi_og.copy()

negp_og2=negp_og.dropna(dim="diameter", how="all")
negp=negp_og2.dropna(dim="time", how="all")

negi_og2=negi_og.dropna(dim="diameter", how="all")
negi=negi_og2.dropna(dim="time", how="all")

neg_p=negp.to_pandas()
neg_i=negi.to_pandas()

common_index = neg_p.index.intersection(neg_i.index)

neg_p = neg_p.loc[common_index]
neg_i = neg_i.loc[common_index]

neg_p.columns = neg_p.columns.astype(float)
neg_i.columns = neg_i.columns.astype(float)

neg_p.index = neg_p.index.tz_localize("UTC").tz_convert("Asia/Shanghai")
neg_i.index = neg_i.index.tz_localize("UTC").tz_convert("Asia/Shanghai")

# Preserve the complete original NAIS particle distribution
neg_p_nais_original = neg_p.copy()

# Preserve the complete original NAIS particle distribution
neg_p_nais_original = neg_p.copy()
neg_i_1 = neg_i.copy()

# Open and sort the DMPS dataset
dmps_df = pd.read_excel("/home/kivekia/Python/DMPS_BUCT_to_D_concentration.xlsx",
                        index_col=0,
                        parse_dates=True)

# DMPS timestamps are already Beijing local time
dmps_df.index = pd.to_datetime(dmps_df.index)

if dmps_df.index.tz is None:
    dmps_df.index = dmps_df.index.tz_localize("Asia/Shanghai")

dmps_df = dmps_df[
    ~dmps_df.index.duplicated(keep="first")]

dmps_df = dmps_df.sort_index()

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Convert NAIS particle and ion distributions
nais_particles_dN = af.dndlogdp2dn(
    neg_p_nais_original.sort_index(axis=1)
)

nais_ions_dN = af.dndlogdp2dn(
    neg_i.sort_index(axis=1)
)

# Convert DMPS distribution
dmps_dN = af.dndlogdp2dn(
    dmps_df.sort_index(axis=1)
)

# --------------------------------------------------
# Interpolate DMPS in time to NAIS timestamps
# --------------------------------------------------

dmps_dN_10min = dmps_dN.reindex(
    nais_particles_dN.index.union(dmps_dN.index)
).sort_index()

dmps_dN_10min = dmps_dN_10min.interpolate(
    method="time",
    limit_area="inside"
)

dmps_dN_10min = dmps_dN_10min.reindex(
    nais_particles_dN.index
)

# --------------------------------------------------
# Interpolate DMPS onto the NAIS diameter grid
# --------------------------------------------------

# NAIS diameter grid (metres)
nais_dp = nais_particles_dN.columns.to_numpy(dtype=float)

# DMPS diameter grid (nm)
dmps_dp = dmps_dN_10min.columns.to_numpy(dtype=float)

# Start with NAIS particle distribution in dN
combined_dN = nais_particles_dN.copy()

# Select NAIS diameters between 10 and 40 nm
dmps_mask = (
    (nais_dp >= 10e-9) &
    (nais_dp <= 40e-9)
)

# Interpolate DMPS dN onto the NAIS diameter grid
for timestamp in nais_particles_dN.index:

    values = dmps_dN_10min.loc[timestamp].to_numpy(dtype=float)

    valid = (
        np.isfinite(values) &
        np.isfinite(dmps_dp) &
        (dmps_dp > 0)
    )

    if valid.sum() < 2:
        combined_dN.loc[timestamp, nais_dp[dmps_mask]] = np.nan
        continue

    combined_dN.loc[timestamp, nais_dp[dmps_mask]] = np.interp(
        np.log10(nais_dp[dmps_mask] * 1e9),
        np.log10(dmps_dp[valid]),
        values[valid],
        left=np.nan,
        right=np.nan
    )

# Use the combined particle distribution and converted NAIS ions
neg_p = combined_dN.copy()
neg_i = nais_ions_dN.copy()

# Remove negative concentrations
neg_i[neg_i < 0] = np.nan
neg_p[neg_p < 0] = np.nan

del dmps_dN_10min
del dmps_dN

# Open data for coagulation calculations
df = pd.read_excel("/home/kivekia/Python/BUCT DMPS 2018-2022/BUCT DMPS 2018-2022/DMPS_BUCT_to_D.xlsx",
                    index_col=0,
                    parse_dates=True
                    )

# DMPS timestamps are already Beijing local time
df.index = pd.to_datetime(df.index)

if df.index.tz is None:
    df.index = df.index.tz_localize("Asia/Shanghai")

df = df[
    ~df.index.duplicated(keep="first")]

df = df.sort_index()


df_10min = df.reindex(
    nais_particles_dN.index.union(df.index)
).sort_index()

df_10min = df_10min.interpolate(
    method="time",
    limit_area="inside"
)

df_10min = df_10min.reindex(
    nais_particles_dN.index
)

# Particle and ion concentrations on identical NAIS diameters
neg_p_charge = neg_p.copy()
neg_i_charge = neg_i.copy()

neg_i_charge[neg_i_charge < 0] = np.nan
neg_p_charge[neg_p_charge < 0] = np.nan

dp_nm = neg_i.columns.to_numpy(dtype=float)
dp_nm_2 = dp_nm * 1e9
df_10min.columns = df_10min.columns.astype(float) * 1e-9

coagN = af.calc_coags(df_10min, dp_nm)
coagS = ((1+1.5/np.sqrt(dp_nm_2)-4/(dp_nm_2+2)**2)*coagN)

N = af.calc_conc(neg_i_1, 0.0, 1.6e-9).squeeze()
if N.index.tz is None:
    N.index = N.index.tz_localize("Asia/Shanghai")
N = N.reindex(nais_particles_dN.index)

neg_p_charge = neg_p_charge.loc[:, neg_p_charge.columns >= 2.5e-9]
neg_i_charge = neg_i_charge.loc[:, neg_i_charge.columns >= 2.5e-9]

concentrations = af.calc_conc_interp(neg_p_nais_original, 4e-9, 6e-9).squeeze()

threshold = 1000

mask = concentrations > threshold

neg_p_masked = neg_p_charge.copy()
neg_i_masked = neg_i_charge.copy()

neg_p_masked.loc[~mask, :] = np.nan
neg_i_masked.loc[~mask, :] = np.nan

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

a = 1.6e-6
d0 = 1e-9
a0 = 0.006e-6
l = 1.2

f_meas = neg_i_charge / neg_p_charge

dp_nm = neg_p_charge.columns.to_numpy(dtype=float)

f_eq_1 = pd.Series(
    {
        diameter: al.eq_charge_frac(diameter, -1)
        for diameter in neg_p.columns
    }
)

f_eq_2 = pd.Series(
    {
        diameter: (a0*(diameter / d0)**l)/a
        for diameter in dp_nm
    }
)

f_eq_3 = pd.DataFrame(
    index=neg_p_charge.index,
    columns=dp_nm,
    dtype=float
)

for diameter in dp_nm:

    A = a0 * (diameter / d0)**l

    f_eq_3[diameter] = (
        A * N
    ) / (
        a * N
        + coagS[diameter]
        - coagN[diameter]
    )

f_eq = f_eq_1 - f_eq_2 + f_eq_3

overcharge = f_meas.divide(
    f_eq,
    axis=1
)

start = "2022-03-13 00:00:00"
end = "2022-03-20 23:59:59"

overcharge = overcharge.loc[start:end]

overcharge_30_min = overcharge.resample("1h").median()

S_small = overcharge.loc[
    :,
    (overcharge.columns >= 2.5e-9) &
    (overcharge.columns <= 4e-9)
]

S_small_median = S_small.median(axis=1)

S_large = overcharge_30_min.loc[
    :,
    (overcharge_30_min.columns >= 20e-9) &
    (overcharge_30_min.columns <= 25e-9)
]

S_large_median = S_large.median(axis=1)

concentrations_plotting = af.calc_conc(neg_p_nais_original, 4e-9, 6e-9).squeeze()
concentrations_plotting = concentrations_plotting.loc[start:end]
concentrations_plotting = concentrations_plotting.resample("1h").median()

if isinstance(concentrations_plotting, pd.DataFrame):
    concentrations_plotting = concentrations_plotting.iloc[:, 0]

# ==================================================
# 1. Detect significant concentration increases
# ==================================================

threshold = 1000

min_increase = 100       # cm-3 over 3 hours
trend_hours = 3

N = concentrations_plotting.copy().sort_index()
S = S_large_median.copy().sort_index()

# Put both time series on the same hourly index
data = pd.concat(
    [N.rename("N"), S.rename("S")],
    axis=1
)

# Concentration increase over the previous 3 hours
data["N_increase"] = data["N"].diff(trend_hours)

# Average hourly slope over the previous 3 hours
data["N_slope"] = data["N_increase"] / trend_hours

# Identify downward-to-upward threshold crossings
data["crossing"] = (
    (data["N"] > threshold) &
    (data["N"].shift(1) <= threshold)
)

# Require a sufficiently large increase and positive slope
data["significant"] = (
    data["crossing"] &
    (data["N_increase"] >= min_increase) &
    (data["N_slope"] > 0)
)

events = data.index[data["significant"]].tolist()

print(f"Number of significant events: {len(events)}")

print("\nEvent times and concentrations:")
print(data.loc[events, ["N", "N_increase", "N_slope", "S"]])


# ==================================================
# 2. Calculate the derivative of S
# ==================================================

# Derivative with respect to time, in hours
dt_hours = (
    data.index.to_series()
    .diff()
    .dt.total_seconds() / 3600
)

data["dS_dt"] = data["S"].diff() / dt_hours

# Optional: relative rate of change (% per hour)
data["relative_dS_dt"] = (
    data["dS_dt"] / data["S"].shift(1) * 100
)


# ==================================================
# 3. Extract S around each concentration event
# ==================================================

before = 6
after = 12

event_results = []

for i, event in enumerate(events):

    window = data.loc[
        event - pd.Timedelta(hours=before):
        event + pd.Timedelta(hours=after)
    ].copy()

    window["event"] = i + 1

    window["relative_hour"] = (
        (window.index - event).total_seconds() / 3600
    )

    event_results.append(window)

if event_results:

    event_data = pd.concat(event_results)

    # Median response across all events
    event_median = (
        event_data
        .groupby("relative_hour")
        [["N", "S", "dS_dt", "relative_dS_dt"]]
        .median()
    )

    print("\nMedian S and derivative around events:")
    print(event_median)

else:
    print("No significant events found.")


# ==================================================
# 4. Plot individual event responses
# ==================================================

if event_results:

    fig, axes = plt.subplots(
        3, 1,
        figsize=(12, 11),
        sharex=True
    )

    for i, window in enumerate(event_results):

        label = f"Event {i+1}"

        axes[0].plot(
            window["relative_hour"],
            window["N"],
            alpha=0.5,
            label=label
        )

        axes[1].plot(
            window["relative_hour"],
            window["S"],
            alpha=0.5,
            label=label
        )

        axes[2].plot(
            window["relative_hour"],
            window["dS_dt"],
            alpha=0.5,
            label=label
        )

    # Median response
    axes[0].plot(
        event_median.index,
        event_median["N"],
        color="black",
        linewidth=2.5,
        label="Median"
    )

    axes[1].plot(
        event_median.index,
        event_median["S"],
        color="black",
        linewidth=2.5,
        label="Median"
    )

    axes[2].plot(
        event_median.index,
        event_median["dS_dt"],
        color="black",
        linewidth=2.5,
        label="Median"
    )

    # Mark threshold-crossing time
    for ax in axes:
        ax.axvline(0, color="red", linestyle="--")
        ax.grid(alpha=0.3)
        ax.legend()

    axes[0].set_ylabel("N (cm$^{-3}$)")
    axes[0].set_title("Particle concentration around threshold crossings")

    axes[1].set_ylabel("S (20–25 nm)")
    axes[1].set_title("Overcharge ratio")

    axes[2].set_ylabel("dS/dt (hour$^{-1}$)")
    axes[2].set_xlabel("Hours relative to threshold crossing")
    axes[2].set_title("Rate of change of S")

    plt.tight_layout()
    plt.savefig("examining_concentration", bbox_inches='tight', dpi=300)

periods = {
    "Before": (-6, -1),
    "During": (0, 3),
    "After": (4, 9)
}

period_results = []

for i, window in enumerate(event_results):

    row = {"Event": i + 1}

    for name, (t1, t2) in periods.items():

        subset = window.loc[
            (window["relative_hour"] >= t1) &
            (window["relative_hour"] <= t2)
        ]

        row[f"S_{name}"] = subset["S"].median()
        row[f"dSdt_{name}"] = subset["dS_dt"].median()

    period_results.append(row)

period_df = pd.DataFrame(period_results)

print(period_df.round(4))