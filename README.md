# The Siren Gap
### Mapping Outdoor Tornado Warning Siren Dead Zones in Oklahoma

---

## Research Question

**Which residential areas in the Moore / Norman / south Oklahoma City corridor fall outside the effective audible range of outdoor warning sirens, and which communities are most vulnerable due to the compounding effects of mobile home prevalence, elderly population, and historical tornado frequency?**

---

## Study Area

**Cleveland County + southern Oklahoma County, Oklahoma**
- Focal communities: Moore, Norman, Oklahoma City (south)
- Bounding box: `west=-97.7, south=35.0, east=-97.1, north=35.6`
- Coordinate reference: UTM Zone 14N (EPSG:32614) for analysis; WGS84 for output

This corridor was chosen because of its exceptional tornado history—including the 1999 Bridge Creek–Moore EF5, the 2013 Moore EF5, and numerous significant events in between—and its mix of dense suburban development and mobile home communities that are particularly vulnerable.

---

## Data Sources

| Dataset | Source | Description |
|---------|--------|-------------|
| Siren Locations | Custom scraped (300 points) | OKC, Moore, Norman, Newcastle, Noble warning sirens |
| USGS 1/3 arc-second DEM | [USGS National Map](https://apps.nationalmap.gov/downloader/) | `n36w098.tif` — elevation for lat 35–36 |
| NOAA SPC Tornado Tracks | [NOAA Storm Prediction Center](https://www.spc.noaa.gov/gis/svrgis/) | 1950–2023 tornado paths, EPSG:4326 |
| ACS B25024 (Housing Units) | [US Census Bureau](https://data.census.gov/) | Mobile home counts, Cleveland + Oklahoma County block groups |
| ACS B01001 (Age) | [US Census Bureau](https://data.census.gov/) | Population by age (65+), same counties |
| TIGER Block Groups | [US Census TIGER/Line 2022](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) | Oklahoma block group boundaries |
| NLCD 2021 | [MRLC](https://www.mrlc.gov/) | Land cover classification, clipped to study area |

---

## Methodology

### 1. Data Preprocessing (`scripts/01_preprocess_data.py`)
- Filters sirens, tornado tracks, and block groups to study bounding box
- Merges ACS B25024 (mobile homes) and B01001 (elderly population) with TIGER block group geometries
- Computes `pct_mobile_home` and `pct_elderly` per block group
- Clips USGS DEM to study area using rasterio windowed read
- Counts tornado track intersections per block group (1993–2023)

### 2. Acoustic Propagation Model (`scripts/02_acoustic_model.py`)
Physics-informed siren audibility model:
- **Source level**: 123 dB at 30 m reference distance
- **Distance decay**: `SPL = SPL_ref - 20·log10(d / d_ref)` (inverse-square law)
- **Ground absorption by NLCD class**: 0 dB/100m (impervious) → 1.0 (crops) → 1.5 (grassland) → 2.0 dB/100m (forest)
- **Terrain shadowing**: DEM line-of-sight check via vectorized ray-sampling; +8 dB attenuation if blocked
- Grid resolution: 50 m (vectorized numpy, no pixel loops)
- Composite: maximum SPL across all sirens at each grid cell
- Dead zones: cells where composite SPL < 70 dB, vectorized to polygons

### 3. Vulnerability Analysis (`scripts/03_vulnerability_analysis.py`)
**Siren Gap Vulnerability Score (SGVS):**

```
SGVS = 0.35·norm(pop_in_dead_zone)
     + 0.30·norm(pct_mobile_home)
     + 0.20·norm(pct_elderly)
     + 0.15·norm(tornado_frequency)
```

Where `norm(x) = (x − min) / (max − min)`.

- Spatial overlay to compute % of each block group area inside dead zones
- Getis-Ord Gi* hot spot analysis (PySAL / esda) to identify spatial clusters of high vulnerability

### 4. Siren Placement Optimization (`scripts/04_siren_optimization.py`)
Greedy **Maximal Covering Location Problem (MCLP)**:
- Candidate locations: 500 m grid inside/near dead zones
- Demand points: block group centroids weighted by `pop_in_dead_zone`
- Coverage radius: 1,500 m (70 dB threshold on Oklahoma flat terrain)
- Scenarios: N = 3, 5, 10 new sirens
- Reports additional population covered, mobile homes, and estimated cost

---

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

> **Note (Windows):** If you have PostgreSQL with PostGIS installed, the PROJ data path may conflict with rasterio. The scripts automatically set:
> ```python
> os.environ['PROJ_DATA'] = 'C:/Program Files/PostgreSQL/17/share/contrib/postgis-3.5/proj'
> os.environ['GDAL_DATA'] = ''
> ```
> Adjust this path in each script if your PostgreSQL installation differs.

### Run Full Pipeline
```bash
cd "c:/Users/kalch/OneDrive/Desktop/Portfolio/Siren Gap"
python scripts/05_run_all.py
```

### Run Individual Scripts
```bash
python scripts/01_preprocess_data.py      # ~2–5 min (DEM clipping)
python scripts/02_acoustic_model.py       # ~5–15 min (acoustic model)
python scripts/03_vulnerability_analysis.py
python scripts/04_siren_optimization.py
```

### View the Web Map
Open `web/index.html` in a browser (use a local server for fetch() to work):
```bash
cd "c:/Users/kalch/OneDrive/Desktop/Portfolio/Siren Gap"
python -m http.server 8080
# then open: http://localhost:8080/web/
```

---

## Key Findings

Analysis complete. Key results from the physics-informed acoustic model:

- **Dead zones**: 486 distinct polygons covering **2,236 km²** fall below the 70 dB audibility threshold across Cleveland and Oklahoma Counties
- **Population exposure**: **246,150 residents (27.1% of the 907,519 study-area population)** live in siren dead zones — areas where outdoor warning sirens cannot reliably be heard
- **Vulnerable households**: **6,349 mobile home units** and **39,407 elderly residents (65+)** are located within dead zones, compounding exposure risk
- **Tornado history**: **126 tornado tracks from 1993–2023** intersect the study area; 26 block groups are identified as Getis-Ord Gi* hot spots for siren vulnerability
- **Optimal placements**: Adding just **5 strategically-placed sirens ($150,000)** would extend warning coverage to an additional **53,979 residents** — a 21.9% improvement — prioritizing mobile home communities
- **10-siren scenario**: Ten new sirens ($300,000) covers an additional **85,146 residents** (34.6% improvement) with the highest-priority sites concentrated in outer south OKC and west Norman

---

## Directory Structure

```
Siren Gap/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── dem/                    USGS DEM tiles
│   │   ├── census/                 ACS + TIGER block groups
│   │   ├── sirens/                 Scraped siren GeoJSON
│   │   ├── tornado_tracks/         NOAA SPC shapefile
│   │   └── nlcd/                   NLCD 2021 land cover
│   └── processed/
│       ├── dem_clip.tif            Clipped DEM (study bbox)
│       ├── dem_utm50.tif           DEM reprojected to UTM 50m
│       └── nlcd_utm50.tif          NLCD reprojected to UTM 50m
├── scripts/
│   ├── 01_preprocess_data.py
│   ├── 02_acoustic_model.py
│   ├── 03_vulnerability_analysis.py
│   ├── 04_siren_optimization.py
│   └── 05_run_all.py
├── outputs/
│   ├── rasters/
│   │   └── spl_composite.tif       Composite SPL surface (dB)
│   └── vectors/
│       ├── sirens_study_area.geojson
│       ├── block_groups_processed.geojson
│       ├── tornado_tracks_study.geojson
│       ├── dead_zones.geojson
│       ├── vulnerability_scores.geojson
│       ├── proposed_sirens_n3.geojson
│       ├── proposed_sirens_n5.geojson
│       └── proposed_sirens_n10.geojson
└── web/
    └── index.html                  Interactive Leaflet map
```

---

## References

- NOAA Storm Prediction Center. (2024). *Severe Weather GIS Data: Tornado Tracks 1950–2023.*
- U.S. Census Bureau. (2022). *American Community Survey 5-Year Estimates, Tables B25024, B01001.*
- U.S. Geological Survey. (2023). *National Elevation Dataset 1/3 Arc-Second.*
- Multi-Resolution Land Characteristics Consortium. (2021). *National Land Cover Database.*
- FEMA. (2023). *Mobile Homes and Tornado Risk.*
- ISO 9613-2. (1996). *Acoustics – Attenuation of sound during propagation outdoors.*
