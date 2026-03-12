"""
02_acoustic_model.py
The Siren Gap - Physics-Informed Acoustic Propagation Model

Models siren sound propagation using:
  - Inverse-square-law distance decay
  - NLCD land-cover ground absorption
  - DEM terrain line-of-sight shadowing

Outputs:
  outputs/rasters/spl_composite.tif  - composite SPL raster (dB)
  outputs/vectors/dead_zones.geojson - polygons where SPL < 70 dB
"""

import os
os.environ['PROJ_DATA'] = 'C:/Users/kalch/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0/LocalCache/local-packages/Python313/site-packages/rasterio/proj_data'
os.environ['GDAL_DATA'] = ''

import math
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject, calculate_default_transform
from rasterio.features import shapes
import rasterio.crs
from shapely.geometry import shape
from pathlib import Path

ROOT    = Path(__file__).parent.parent
DATA_PROC = ROOT / 'data' / 'processed'
OUT_VEC = ROOT / 'outputs' / 'vectors'
OUT_RAS = ROOT / 'outputs' / 'rasters'

for d in [OUT_VEC, OUT_RAS]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SPL_REF   = 123.0   # dB at d_ref
D_REF     = 30.0    # metres
THRESHOLD = 70.0    # dB audibility threshold
TERRAIN_SHADOW_DB = 8.0  # extra attenuation if terrain-shadowed
GRID_RES  = 50      # metres (computation grid)
UTM14N    = 'EPSG:32614'

# Ground absorption by NLCD class (dB per 100 m)
NLCD_ABSORPTION = {
    # Open water / ice
    11: 0.0, 12: 0.0,
    # Developed
    21: 0.0, 22: 0.0, 23: 0.0, 24: 0.0,
    # Barren
    31: 0.5,
    # Forest
    41: 2.0, 42: 2.0, 43: 2.0,
    # Shrub/scrub
    52: 1.5,
    # Grassland
    71: 1.5,
    # Pasture / hay
    81: 1.0,
    # Cultivated crops
    82: 1.0,
    # Wetlands
    90: 1.0, 95: 1.0,
}
DEFAULT_ABSORPTION = 1.0  # dB/100m for unmapped classes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reproject_raster_to_utm(src_path, dst_path, resolution=50):
    """Reproject a raster to UTM 14N at given resolution (metres)."""
    with rasterio.open(src_path) as src:
        dst_crs = rasterio.crs.CRS.from_epsg(32614)
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds,
            resolution=resolution
        )
        profile = src.profile.copy()
        profile.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'driver': 'GTiff',
            'compress': 'lzw',
            'nodata': src.nodata if (src.nodata is not None and src.nodata >= np.iinfo(np.uint8).min and src.nodata <= np.iinfo(np.uint8).max) else (0 if str(profile.get('dtype','')) == 'uint8' else -9999),
        })
        with rasterio.open(dst_path, 'w', **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear
            )
    return dst_path


def build_absorption_grid(nlcd_data, nlcd_nodata):
    """Map NLCD class codes to absorption rate (dB/100m) array."""
    absorb = np.full(nlcd_data.shape, DEFAULT_ABSORPTION, dtype=np.float32)
    for code, rate in NLCD_ABSORPTION.items():
        absorb[nlcd_data == code] = rate
    if nlcd_nodata is not None:
        absorb[nlcd_data == nlcd_nodata] = DEFAULT_ABSORPTION
    return absorb


def check_terrain_shadow_vectorized(dem, transform, siren_row, siren_col, siren_elev,
                                    sample_steps=8):
    """
    For every grid cell, check terrain line-of-sight to siren using
    vectorized numpy sampling along the path.

    Returns boolean array: True = shadowed (blocked by terrain).
    """
    nrows, ncols = dem.shape
    rows, cols = np.mgrid[0:nrows, 0:ncols]

    # Number of intermediate samples along the ray
    n = sample_steps
    shadow = np.zeros((nrows, ncols), dtype=bool)

    # Fractional step positions (exclude 0 and 1 to avoid checking siren/target itself)
    fracs = np.linspace(0.0, 1.0, n + 2)[1:-1]  # shape (n,)

    for t in fracs:
        # Interpolated row/col along ray from siren to each pixel
        interp_row = (siren_row * (1 - t) + rows * t).astype(int)
        interp_col = (siren_col * (1 - t) + cols * t).astype(int)

        # Clamp to valid range
        interp_row = np.clip(interp_row, 0, nrows - 1)
        interp_col = np.clip(interp_col, 0, ncols - 1)

        # Elevation at interpolated point
        terrain_elev = dem[interp_row, interp_col].astype(float)

        # Expected elevation along straight siren->target line at this fraction
        target_elev = dem[rows, cols].astype(float)
        expected_elev = siren_elev * (1 - t) + target_elev * t

        # Mark as shadowed if terrain pokes above the straight line
        shadow |= (terrain_elev > expected_elev + 2.0)   # 2 m tolerance

    return shadow


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[1/7] Reprojecting DEM to UTM 14N (50 m grid)...")
    dem_clip = DATA_PROC / 'dem_clip.tif'
    dem_utm  = DATA_PROC / 'dem_utm50.tif'
    reproject_raster_to_utm(dem_clip, dem_utm, resolution=GRID_RES)
    print(f"      DEM reprojected -> {dem_utm}")

    print("[2/7] Reprojecting NLCD to UTM 14N (50 m grid)...")
    nlcd_src = ROOT / 'data' / 'raw' / 'nlcd' / 'nlcd_2021_ok_geo.tif'
    nlcd_utm = DATA_PROC / 'nlcd_utm50.tif'
    reproject_raster_to_utm(nlcd_src, nlcd_utm, resolution=GRID_RES)
    print(f"      NLCD reprojected -> {nlcd_utm}")

    print("[3/7] Loading grids and sirens...")
    with rasterio.open(dem_utm) as dem_ds:
        dem_data  = dem_ds.read(1).astype(float)
        dem_transform = dem_ds.transform
        dem_nodata    = dem_ds.nodata
        dem_crs       = dem_ds.crs
        nrows, ncols  = dem_data.shape
        dem_profile   = dem_ds.profile.copy()

    # Replace nodata in DEM with mean elevation
    if dem_nodata is not None:
        mask_nd = (dem_data == dem_nodata)
        mean_elev = dem_data[~mask_nd].mean() if mask_nd.any() else dem_data.mean()
        dem_data[mask_nd] = mean_elev
    print(f"      DEM shape: {dem_data.shape}, elev range: {dem_data.min():.0f}-{dem_data.max():.0f} m")

    with rasterio.open(nlcd_utm) as nlcd_ds:
        # Resample NLCD to match DEM grid exactly
        nlcd_data = nlcd_ds.read(
            1,
            out_shape=(nrows, ncols),
            resampling=Resampling.nearest
        ).astype(np.int16)
        nlcd_nodata = nlcd_ds.nodata

    absorb_grid = build_absorption_grid(nlcd_data, nlcd_nodata)
    print(f"      NLCD absorption grid built, mean rate: {absorb_grid.mean():.2f} dB/100m")

    # Load sirens (already in UTM)
    sirens = gpd.read_file(OUT_VEC / 'sirens_study_area.geojson')
    print(f"      Sirens loaded: {len(sirens)}")

    # Pixel coordinate helper
    inv_transform = ~dem_transform

    def lonlat_to_rowcol(x, y):
        col, row = inv_transform * (x, y)
        return int(np.clip(row, 0, nrows - 1)), int(np.clip(col, 0, ncols - 1))

    # Build pixel-centre coordinate arrays (metres, UTM)
    pixel_res = dem_transform.a   # metres per pixel (positive)
    xs = dem_transform.c + (np.arange(ncols) + 0.5) * pixel_res
    ys = dem_transform.f + (np.arange(nrows) + 0.5) * dem_transform.e  # e is negative

    X, Y = np.meshgrid(xs, ys)   # shape (nrows, ncols)

    # Composite SPL: start with very low value everywhere
    spl_composite = np.full((nrows, ncols), -999.0, dtype=np.float32)

    print(f"[4/7] Computing acoustic propagation for {len(sirens)} sirens...")

    for idx, siren in sirens.iterrows():
        sx, sy = siren.geometry.x, siren.geometry.y
        siren_row, siren_col = lonlat_to_rowcol(sx, sy)
        siren_elev = float(dem_data[siren_row, siren_col])
        spl_ref    = float(siren.get('spl_at_30m', SPL_REF))

        # Distance from siren to every grid cell
        dx = X - sx
        dy = Y - sy
        dist = np.sqrt(dx**2 + dy**2)
        dist = np.maximum(dist, D_REF)   # clamp to reference distance

        # Inverse-square-law decay
        spl = spl_ref - 20.0 * np.log10(dist / D_REF)

        # Ground absorption: absorb_grid is in dB/100m
        absorption_total = absorb_grid * (dist / 100.0)
        spl -= absorption_total

        # Terrain shadowing (vectorized)
        shadow = check_terrain_shadow_vectorized(
            dem_data, dem_transform,
            siren_row, siren_col, siren_elev,
            sample_steps=6
        )
        spl[shadow] -= TERRAIN_SHADOW_DB

        # Update composite (take maximum at each cell)
        spl_composite = np.maximum(spl_composite, spl)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(sirens):
            print(f"      Processed {idx + 1}/{len(sirens)} sirens ...")

    print("[5/7] Saving SPL composite raster...")
    spl_profile = dem_profile.copy()
    spl_profile.update({
        'dtype': 'float32',
        'count': 1,
        'nodata': -9999.0,
        'compress': 'lzw',
        'driver': 'GTiff',
    })
    out_spl = OUT_RAS / 'spl_composite.tif'
    with rasterio.open(out_spl, 'w', **spl_profile) as dst:
        dst.write(spl_composite.astype(np.float32), 1)
    print(f"      SPL range: {spl_composite.max():.1f} dB (max) | {spl_composite.min():.1f} dB (min)")
    print(f"      Saved -> outputs/rasters/spl_composite.tif")

    print("[6/7] Vectorizing dead zones (SPL < 70 dB)...")
    # Dead zone: SPL below threshold (and valid data)
    dead_mask = (spl_composite < THRESHOLD).astype(np.uint8)

    # Convert to polygons
    polys = []
    for geom, val in shapes(dead_mask, transform=dem_transform):
        if val == 1:  # dead zone pixels
            polys.append(shape(geom))

    if len(polys) == 0:
        print("      WARNING: No dead zone polygons found. Check SPL values.")
        # Create empty GeoDataFrame
        dz_gdf = gpd.GeoDataFrame({'area_m2': []}, geometry=[], crs=UTM14N)
    else:
        dz_gdf = gpd.GeoDataFrame(
            {'area_m2': [p.area for p in polys]},
            geometry=polys,
            crs=UTM14N
        )
        # Dissolve adjacent cells into larger polygons
        dz_gdf = dz_gdf.dissolve().explode(index_parts=False).reset_index(drop=True)
        dz_gdf['area_km2'] = dz_gdf.geometry.area / 1e6
        # Filter out tiny slivers (< 0.01 km2)
        dz_gdf = dz_gdf[dz_gdf['area_km2'] >= 0.01].reset_index(drop=True)

    print(f"      Dead zone polygons: {len(dz_gdf)}")
    if len(dz_gdf) > 0:
        total_dz_km2 = dz_gdf['area_km2'].sum()
        print(f"      Total dead zone area: {total_dz_km2:.1f} km²")

    dz_gdf.to_file(OUT_VEC / 'dead_zones.geojson', driver='GeoJSON')
    print("      Saved -> outputs/vectors/dead_zones.geojson")

    print("[7/7] Acoustic model complete.")
    cells_covered = (spl_composite >= THRESHOLD).sum()
    total_cells   = spl_composite.size
    pct_covered   = 100.0 * cells_covered / total_cells
    print(f"      Coverage at >=70 dB: {pct_covered:.1f}% of study area")
    print(f"      Dead zone area    : {100 - pct_covered:.1f}% of study area")


if __name__ == '__main__':
    main()
