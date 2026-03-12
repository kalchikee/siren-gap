"""
01_preprocess_data.py
The Siren Gap - Data Preprocessing

Loads, filters, reprojects, and joins all raw data sources.
Outputs processed GeoJSONs and a clipped DEM for downstream scripts.
"""

import os
# PROJ fix for PostgreSQL conflict - must be set before rasterio import
os.environ['PROJ_DATA'] = 'C:/Program Files/PostgreSQL/17/share/contrib/postgis-3.5/proj'
os.environ['GDAL_DATA'] = ''

import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.windows
from rasterio.windows import from_bounds
from pathlib import Path
from shapely.geometry import box

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(__file__).parent.parent
DATA_RAW  = ROOT / 'data' / 'raw'
DATA_PROC = ROOT / 'data' / 'processed'
OUT_VEC   = ROOT / 'outputs' / 'vectors'
OUT_RAS   = ROOT / 'outputs' / 'rasters'

# Study area bounding box (EPSG:4326)
BBOX      = (-97.7, 35.0, -97.1, 35.6)   # west, south, east, north
STUDY_BOX = box(*BBOX)

UTM14N = 'EPSG:32614'
WGS84  = 'EPSG:4326'


# ---------------------------------------------------------------------------
# Census helpers (defined at module level so they are available on import)
# ---------------------------------------------------------------------------

def parse_census(filepath):
    """Parse ACS JSON (list-of-lists format) into a DataFrame."""
    with open(filepath) as f:
        rows = json.load(f)
    header = rows[0]
    df = pd.DataFrame(rows[1:], columns=header)
    # Build GEOID matching TIGER format: state+county+tract+bg (12 chars)
    df['GEOID'] = df['state'] + df['county'] + df['tract'] + df['block group']
    return df


def load_b25024(county_code):
    fname = DATA_RAW / 'census' / f'B25024_county{county_code}.json'
    df = parse_census(fname)
    df['total_units']       = pd.to_numeric(df['B25024_001E'], errors='coerce').fillna(0)
    df['mobile_home_units'] = pd.to_numeric(df['B25024_010E'], errors='coerce').fillna(0)
    return df[['GEOID', 'total_units', 'mobile_home_units']]


def load_b01001(county_code):
    fname = DATA_RAW / 'census' / f'B01001_county{county_code}.json'
    df = parse_census(fname)
    elderly_cols_male   = ['B01001_020E', 'B01001_021E', 'B01001_022E',
                           'B01001_023E', 'B01001_024E', 'B01001_025E']
    elderly_cols_female = ['B01001_044E', 'B01001_045E', 'B01001_046E',
                           'B01001_047E', 'B01001_048E', 'B01001_049E']
    all_elderly = elderly_cols_male + elderly_cols_female
    for c in all_elderly + ['B01001_001E']:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    df['total_pop']   = df['B01001_001E']
    df['elderly_pop'] = df[all_elderly].sum(axis=1)
    return df[['GEOID', 'total_pop', 'elderly_pop']]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    for d in [DATA_PROC, OUT_VEC, OUT_RAS]:
        d.mkdir(parents=True, exist_ok=True)

    # ---- 1. Sirens ----
    print("[1/6] Loading and filtering sirens...")
    sirens = gpd.read_file(DATA_RAW / 'sirens' / 'ok_sirens.geojson')
    print(f"      Raw sirens: {len(sirens)}")
    sirens = sirens[sirens.geometry.within(STUDY_BOX)].copy()
    print(f"      Sirens in study bbox: {len(sirens)}")
    sirens = sirens.set_crs(WGS84).to_crs(UTM14N)
    sirens.to_file(OUT_VEC / 'sirens_study_area.geojson', driver='GeoJSON')
    print("      Saved -> outputs/vectors/sirens_study_area.geojson")

    # ---- 2. Census ----
    print("[2/6] Loading census data (mobile homes + elderly)...")
    df_mh  = pd.concat([load_b25024('027'), load_b25024('109')], ignore_index=True)
    df_age = pd.concat([load_b01001('027'), load_b01001('109')], ignore_index=True)

    census_df = df_mh.merge(df_age, on='GEOID', how='outer')
    census_df['pct_mobile_home'] = np.where(
        census_df['total_units'] > 0,
        census_df['mobile_home_units'] / census_df['total_units'] * 100,
        0.0
    )
    census_df['pct_elderly'] = np.where(
        census_df['total_pop'] > 0,
        census_df['elderly_pop'] / census_df['total_pop'] * 100,
        0.0
    )
    print(f"      Census records merged: {len(census_df)}")

    # ---- 3. Block Groups ----
    print("[3/6] Loading block group shapefile...")
    bg = gpd.read_file(DATA_RAW / 'census' / 'tl_2022_40_bg.shp')
    bg = bg[(bg['COUNTYFP'].isin(['027', '109'])) & (bg['STATEFP'] == '40')].copy()
    print(f"      Block groups (Cleveland + Oklahoma Co.): {len(bg)}")

    bg = bg.to_crs(WGS84)
    bg = bg[bg.geometry.intersects(STUDY_BOX)].copy()
    print(f"      Block groups in study bbox: {len(bg)}")

    census_df['GEOID'] = census_df['GEOID'].astype(str).str.zfill(12)
    bg['GEOID']        = bg['GEOID'].astype(str).str.zfill(12)
    bg = bg.merge(census_df, on='GEOID', how='left')

    for col in ['total_units', 'mobile_home_units', 'total_pop', 'elderly_pop',
                'pct_mobile_home', 'pct_elderly']:
        bg[col] = bg[col].fillna(0.0)

    print(f"      Census join complete. NaN pct_mobile_home: {bg['pct_mobile_home'].isna().sum()}")

    # ---- 4. Tornado Tracks ----
    print("[4/6] Loading tornado tracks...")
    tracks = gpd.read_file(DATA_RAW / 'tornado_tracks' / '1950-2023-torn-aspath')
    print(f"      Total tornado records: {len(tracks)}")

    tracks = tracks[(tracks['st'] == 'OK') & (tracks['yr'] >= 1993)].copy()
    print(f"      Oklahoma tracks 1993-2023: {len(tracks)}")

    tracks = tracks[tracks.geometry.intersects(STUDY_BOX)].copy()
    print(f"      Tracks intersecting study bbox: {len(tracks)}")

    print("      Computing tornado density per block group...")
    bg_utm     = bg.to_crs(UTM14N)
    tracks_utm = tracks.to_crs(UTM14N)

    joined = gpd.sjoin(
        tracks_utm[['yr', 'mag', 'geometry']],
        bg_utm[['GEOID', 'geometry']],
        how='inner', predicate='intersects'
    )
    tornado_counts = joined.groupby('GEOID').size().reset_index(name='tornado_count')
    bg = bg.merge(tornado_counts, on='GEOID', how='left')
    bg['tornado_count'] = bg['tornado_count'].fillna(0).astype(int)
    print(f"      Max tornado count per BG: {bg['tornado_count'].max()}")

    bg_out = bg.to_crs(UTM14N)
    bg_out.to_file(OUT_VEC / 'block_groups_processed.geojson', driver='GeoJSON')
    print("      Saved -> outputs/vectors/block_groups_processed.geojson")

    tracks_out = tracks.to_crs(UTM14N)
    tracks_out.to_file(OUT_VEC / 'tornado_tracks_study.geojson', driver='GeoJSON')
    print("      Saved -> outputs/vectors/tornado_tracks_study.geojson")

    # ---- 5. DEM Clip ----
    print("[5/6] Clipping DEM to study area...")
    # n36w098 covers lat 35-36, lon -98 to -97; Moore/Norman are at lat ~35.0-35.6
    dem_path = DATA_RAW / 'dem' / 'n36w098.tif'
    out_dem  = DATA_PROC / 'dem_clip.tif'

    with rasterio.open(dem_path) as src:
        print(f"      DEM source CRS: {src.crs}, shape: {src.shape}")
        win = from_bounds(BBOX[0], BBOX[1], BBOX[2], BBOX[3], src.transform)
        win = win.round_offsets().round_lengths()

        row_off = max(0, int(win.row_off))
        col_off = max(0, int(win.col_off))
        row_end = min(src.height, int(win.row_off + win.height))
        col_end = min(src.width,  int(win.col_off + win.width))
        win_clamped = rasterio.windows.Window(
            col_off, row_off,
            col_end - col_off,
            row_end - row_off
        )

        data          = src.read(1, window=win_clamped)
        win_transform = src.window_transform(win_clamped)

        profile = src.profile.copy()
        profile.update({
            'height':    data.shape[0],
            'width':     data.shape[1],
            'transform': win_transform,
            'driver':    'GTiff',
            'compress':  'lzw',
        })

        with rasterio.open(out_dem, 'w', **profile) as dst:
            dst.write(data, 1)

    print(f"      Clipped DEM shape: {data.shape}  -> data/processed/dem_clip.tif")

    # ---- 6. Summary ----
    print("[6/6] Preprocessing complete.")
    print(f"      Sirens in study area : {len(sirens)}")
    print(f"      Block groups         : {len(bg_out)}")
    print(f"      Tornado tracks       : {len(tracks_out)}")
    print(f"      DEM clipped shape    : {data.shape}")


if __name__ == '__main__':
    main()
