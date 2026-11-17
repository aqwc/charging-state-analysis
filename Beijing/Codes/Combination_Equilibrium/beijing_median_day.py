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

# Change the index to 1-hour intervals
neg_i_raw = neg_i.copy()
neg_p_raw = neg_p.copy()

neg_i_hourly = neg_i_raw.resample("1h").median()
neg_p_hourly = neg_p_raw.resample("1h").median()

neg_i_hourly = neg_i_hourly.dropna(how="all")
neg_p_hourly = neg_p_hourly.dropna(how="all")

neg_p_nais_hourly = (
    nais_particles_dN
    .resample("1h")
    .median()
    .dropna(how="all")
)

common_index = (
    neg_i_hourly.index
    .intersection(neg_p_hourly.index)
    .intersection(neg_p_nais_hourly.index)
)

neg_i_nais = neg_i_hourly.loc[common_index].copy()
neg_p_nais = neg_p_hourly.loc[common_index].copy()

# Particle and ion concentrations on identical NAIS diameters
neg_p_charge = neg_p_nais.copy()
neg_i_charge = neg_i_nais.copy()

neg_i_charge[neg_i_charge < 0] = np.nan
neg_p_charge[neg_p_charge < 0] = np.nan

neg_p_charge = neg_p_charge.loc[:, neg_p_charge.columns >= 2.5e-9]
neg_i_charge = neg_i_charge.loc[:, neg_i_charge.columns >= 2.5e-9]

concentrations = af.calc_conc_interp(neg_p_nais_original, 4e-9, 6e-9).squeeze()

threshold = 1000

mask = concentrations > threshold

neg_p_masked = neg_p_charge.copy()
neg_i_masked = neg_i_charge.copy()

neg_p_masked.loc[~mask, :] = np.nan
neg_i_masked.loc[~mask, :] = np.nan

start = "2020-10-01 00:00:00"
end = "2022-06-29 23:59:59"

neg_p_masked= neg_p_masked.loc[start:end]

df_30 = neg_p_masked

#df_30 = df_30.dropna(how="all")

df_30["clocktime"] = df_30.index.strftime('%H:%M')

median_diurnal = (
    df_30
    .groupby("clocktime")
    .median(numeric_only=True)
)

times = sorted(median_diurnal.index)

median_diurnal = median_diurnal.loc[times]

median_diurnal.index = pd.to_datetime(
    median_diurnal.index,
    format='%H:%M')

#median_diurnal = median_diurnal.loc[:, median_diurnal.columns <= 10e-9]

fig, ax = plt.subplots(figsize=(12,6))

im, cbar = ap.plot_aerosol_dist(
    median_diurnal,
    ax,
    norm='log',

)
#    clim=[0.28, 0.60]
ax.xaxis.set_major_formatter(
    mdates.DateFormatter('%H:%M'))


#cbar.set_ticks([0.30, 0.35, 0.40, 0.45, 0.50, 0.55])
#cbar.set_ticklabels(["0.30", "0.35", "0.40", "0.45", "0.50", "0.55"])

cbar.set_label('Concentration (cm$^{-3}$)')
#plt.show()

plt.savefig("ratio_beijing_concentration_average_day_2", bbox_inches='tight', dpi=300)