import xarray as xr
import numpy as np
import pandas as pd
import aerosol.functions as af
import importlib.metadata
import matplotlib.pyplot as plt
import aerosol.plotting as ap
import glob
import aerosol.lab as al

import matplotlib
import matplotlib.cm as cm

if not hasattr(cm, "get_cmap"):
    cm.get_cmap = matplotlib.colormaps.get_cmap

ds = xr.open_dataset("/home/kivekia/Python/beijing_all.nc")


ds = ds.sortby('time')

neg_p = ds["neg_particles"].to_pandas()
neg_i_1 = ds["neg_ions"].to_pandas()

common_index = neg_p.index.intersection(neg_i_1.index)

neg_p = neg_p.loc[common_index]
neg_i = neg_i_1.loc[common_index]

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

dN = af.calc_conc(neg_p_nais_original, 4e-9, 6e-9).squeeze()

start = "2022-03-14 00:00:00"
end = "2022-03-27 23:59:59"

#start = "2020-10-01 00:00:00"
#end = "2022-06-29 23:59:59"

dN = dN.loc[start:end]
dN_30 = dN.resample("1h").median()

overcharge = overcharge.loc[start:end]
overcharge_4_6nm = overcharge.loc[
    :, (overcharge.columns >= 4e-9) & (overcharge.columns <= 6e-9)
]

overcharge_median = overcharge_4_6nm.median(axis=1)

overcharge_30_min = overcharge_median.resample("1h").median()

overcharge_30min_std = (
    overcharge_median
    .resample("1h")
    .std()
)

# Align the two time series
plot_df = pd.concat(
    [
        dN_30.rename("dN"),
        overcharge_30min_std.rename("overcharge_std")
    ],
    axis=1
).dropna()

# Scatter plot
fig, ax = plt.subplots(figsize=(8, 6))

"""ax.plot(
    plot_df["dN"],
    plot_df["overcharge_std"],
    linestyle="-",
    linewidth=1.2
)"""

ax.scatter(
    plot_df["dN"],
    plot_df["overcharge_std"],
    alpha=0.7,
    s=30
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Particle concentration $dN$ (cm$^{-3}$)")
ax.set_ylabel("Overcharge standard deviation")
ax.set_title("Overcharge variability vs particle concentration (4–6 nm)")

ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig("scatter_std_S_vs_N_of_4_6_nm", bbox_inches='tight', dpi=300)