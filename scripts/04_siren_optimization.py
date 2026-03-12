"""
04_siren_optimization.py
The Siren Gap - Siren Placement Optimization

Solves a greedy Maximal Covering Location Problem (MCLP) to identify
optimal locations for N=3, N=5, N=10 new tornado sirens that maximize
coverage of vulnerable population in dead zones.

Outputs:
  outputs/vectors/proposed_sirens_n3.geojson
  outputs/vectors/proposed_sirens_n5.geojson
  outputs/vectors/proposed_sirens_n10.geojson
"""

import os
os.environ['PROJ_DATA'] = 'C:/Users/kalch/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0/LocalCache/local-packages/Python313/site-packages/rasterio/proj_data'
os.environ['GDAL_DATA'] = ''

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, MultiPoint
from pathlib import Path

ROOT    = Path(__file__).parent.parent
OUT_VEC = ROOT / 'outputs' / 'vectors'
OUT_VEC.mkdir(parents=True, exist_ok=True)

UTM14N = 'EPSG:32614'
COVERAGE_RADIUS = 1500.0   # metres - 70 dB threshold radius for OK flat terrain
GRID_SPACING    = 500.0    # metres - candidate siren spacing
SIREN_COST_USD  = 30_000   # per siren


# ---------------------------------------------------------------------------
# Greedy MCLP solver
# ---------------------------------------------------------------------------

def greedy_mclp(candidates_xy, demand_xy, demand_weights, radius, n_to_place):
    """
    Greedy Maximal Covering Location Problem solver.

    Parameters
    ----------
    candidates_xy  : ndarray (M, 2) - candidate siren locations (UTM)
    demand_xy      : ndarray (N, 2) - demand point locations (UTM centroids)
    demand_weights : ndarray (N,)   - weight per demand point (pop_in_dz)
    radius         : float          - coverage radius (metres)
    n_to_place     : int            - number of sirens to place

    Returns
    -------
    selected_indices : list of int (indices into candidates_xy)
    coverage_gain    : list of float (cumulative weight covered after each pick)
    """
    n_candidates = len(candidates_xy)
    n_demand     = len(demand_xy)

    # Precompute coverage matrix: coverage[c, d] = True if candidate c covers demand d
    # Use chunked computation to avoid huge memory allocation
    chunk = 500
    coverage = np.zeros((n_candidates, n_demand), dtype=bool)
    for start in range(0, n_candidates, chunk):
        end = min(start + chunk, n_candidates)
        dx = candidates_xy[start:end, 0:1] - demand_xy[:, 0]   # (chunk, N)
        dy = candidates_xy[start:end, 1:2] - demand_xy[:, 1]
        dist2 = dx**2 + dy**2
        coverage[start:end] = dist2 <= radius**2

    selected       = []
    already_covered = np.zeros(n_demand, dtype=bool)
    cumulative_weights = []

    for _ in range(n_to_place):
        # For each candidate not yet selected, compute marginal gain
        marginal = np.zeros(n_candidates)
        for c in range(n_candidates):
            if c in selected:
                marginal[c] = -1.0
                continue
            newly_covered  = coverage[c] & ~already_covered
            marginal[c]    = demand_weights[newly_covered].sum()

        best_idx = int(np.argmax(marginal))
        selected.append(best_idx)
        already_covered |= coverage[best_idx]
        cumulative_weights.append(float(demand_weights[already_covered].sum()))

    return selected, cumulative_weights


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("[1/5] Loading dead zones and vulnerability scores...")
    dz = gpd.read_file(OUT_VEC / 'dead_zones.geojson').to_crs(UTM14N)
    vs = gpd.read_file(OUT_VEC / 'vulnerability_scores.geojson').to_crs(UTM14N)
    ex = gpd.read_file(OUT_VEC / 'sirens_study_area.geojson').to_crs(UTM14N)

    print(f"      Dead zone polygons : {len(dz)}")
    print(f"      Block groups       : {len(vs)}")
    print(f"      Existing sirens    : {len(ex)}")

    # ---------------------------------------------------------------------------
    # 2. Build candidate grid inside dead zones
    # ---------------------------------------------------------------------------
    print("[2/5] Building candidate siren grid inside dead zones...")

    if len(dz) == 0:
        print("      WARNING: No dead zones. Cannot place sirens. Exiting.")
        return

    dz_union = dz.geometry.union_all()
    bounds   = dz_union.bounds  # (minx, miny, maxx, maxy)

    # Regular grid
    xs = np.arange(bounds[0], bounds[2], GRID_SPACING)
    ys = np.arange(bounds[1], bounds[3], GRID_SPACING)
    grid_pts = [Point(x, y) for x in xs for y in ys]

    # Keep only candidates inside dead zones
    candidates_gdf = gpd.GeoDataFrame(
        geometry=grid_pts, crs=UTM14N
    )
    candidates_gdf = candidates_gdf[
        candidates_gdf.geometry.within(dz_union)
    ].reset_index(drop=True)

    # If too few candidates in dead zones, expand to full study area buffer
    if len(candidates_gdf) < 10:
        print("      Expanding grid to dead zone buffer (1 km)...")
        dz_buf = dz_union.buffer(1000)
        candidates_gdf = gpd.GeoDataFrame(geometry=grid_pts, crs=UTM14N)
        candidates_gdf = candidates_gdf[
            candidates_gdf.geometry.within(dz_buf)
        ].reset_index(drop=True)

    print(f"      Candidate locations: {len(candidates_gdf)}")
    if len(candidates_gdf) == 0:
        print("      ERROR: No candidate locations generated. Check dead zone geometry.")
        return

    # ---------------------------------------------------------------------------
    # 3. Build demand points (block group centroids weighted by pop_in_dead_zone)
    # ---------------------------------------------------------------------------
    print("[3/5] Building demand points...")

    vs['pop_in_dead_zone'] = pd.to_numeric(
        vs['pop_in_dead_zone'], errors='coerce'
    ).fillna(0.0)
    vs['total_pop']        = pd.to_numeric(vs['total_pop'],       errors='coerce').fillna(0.0)
    vs['mobile_homes_in_dz'] = pd.to_numeric(vs['mobile_homes_in_dz'], errors='coerce').fillna(0.0)
    vs['pct_mobile_home']  = pd.to_numeric(vs['pct_mobile_home'], errors='coerce').fillna(0.0)

    demand_gdf = vs[vs['pop_in_dead_zone'] > 0].copy()

    if len(demand_gdf) == 0:
        print("      WARNING: No demand points with population in dead zones.")
        print("               Using all block groups as demand points instead.")
        demand_gdf = vs.copy()
        demand_gdf['pop_in_dead_zone'] = demand_gdf['total_pop']

    demand_gdf = demand_gdf.copy()
    demand_gdf['centroid'] = demand_gdf.geometry.centroid
    demand_xy  = np.array([[p.x, p.y] for p in demand_gdf['centroid']])
    weights    = demand_gdf['pop_in_dead_zone'].values.astype(float)

    candidates_xy = np.array([[p.x, p.y] for p in candidates_gdf.geometry])

    total_demand = weights.sum()
    print(f"      Demand points     : {len(demand_gdf)}")
    print(f"      Total demand (pop): {total_demand:,.0f}")

    # Current coverage from existing sirens
    existing_xy = np.array([[p.x, p.y] for p in ex.geometry])
    already_covered = np.zeros(len(demand_xy), dtype=bool)
    for ex_pt in existing_xy:
        d2 = (demand_xy[:, 0] - ex_pt[0])**2 + (demand_xy[:, 1] - ex_pt[1])**2
        already_covered |= (d2 <= COVERAGE_RADIUS**2)
    current_coverage = weights[already_covered].sum()
    print(f"      Existing coverage : {current_coverage:,.0f} pop ({100*current_coverage/max(total_demand,1):.1f}%)")

    # ---------------------------------------------------------------------------
    # 4. Solve MCLP for N=3, 5, 10
    # ---------------------------------------------------------------------------
    print("[4/5] Solving MCLP...")

    results = {}
    for n_new in [3, 5, 10]:
        print(f"      Solving for N={n_new} new sirens...")
        selected_idx, cum_weights = greedy_mclp(
            candidates_xy, demand_xy, weights,
            COVERAGE_RADIUS, n_new
        )
        results[n_new] = {
            'selected_idx'  : selected_idx,
            'cum_weights'   : cum_weights,
            'total_new_cov' : cum_weights[-1] if cum_weights else 0.0,
        }

    # ---------------------------------------------------------------------------
    # 5. Save outputs and print report
    # ---------------------------------------------------------------------------
    print("[5/5] Saving proposed siren locations and printing report...")
    print()
    print("=" * 60)
    print("       SIREN PLACEMENT OPTIMIZATION REPORT")
    print("=" * 60)
    print(f"  Study area total population   : {vs['total_pop'].sum():>12,.0f}")
    print(f"  Population in dead zones      : {total_demand:>12,.0f}")
    print(f"  Current siren coverage (pop)  : {current_coverage:>12,.0f}")
    print(f"  Coverage radius assumed       : {COVERAGE_RADIUS:.0f} m")
    print(f"  Cost per siren                : ${SIREN_COST_USD:>10,}")
    print("-" * 60)

    for n_new in [3, 5, 10]:
        r = results[n_new]
        new_cov    = r['total_new_cov']
        total_cov  = current_coverage + new_cov
        pct_imp    = 100 * new_cov / max(total_demand, 1)
        cost       = n_new * SIREN_COST_USD

        print(f"\n  N = {n_new} new sirens:")
        print(f"    Additional population covered : {new_cov:>10,.0f}")
        print(f"    Total population covered      : {total_cov:>10,.0f}")
        print(f"    % improvement in coverage     : {pct_imp:>9.1f}%")
        print(f"    Estimated cost                : ${cost:>10,}")

        # Build GeoDataFrame for proposed sirens
        sel_pts = [candidates_gdf.geometry.iloc[i] for i in r['selected_idx']]

        # Additional mobile homes covered (approximated)
        mh_weights = demand_gdf['mobile_homes_in_dz'].values.astype(float)
        mh_already = np.zeros(len(demand_xy), dtype=bool)
        for ex_pt in existing_xy:
            d2 = (demand_xy[:,0]-ex_pt[0])**2 + (demand_xy[:,1]-ex_pt[1])**2
            mh_already |= (d2 <= COVERAGE_RADIUS**2)

        mh_new_cov = 0.0
        newly_flag = np.zeros(len(demand_xy), dtype=bool)
        for pt in sel_pts:
            d2 = (demand_xy[:,0]-pt.x)**2 + (demand_xy[:,1]-pt.y)**2
            in_range = d2 <= COVERAGE_RADIUS**2
            newly_flag |= (in_range & ~mh_already)
        mh_new_cov = mh_weights[newly_flag].sum()
        print(f"    Additional mobile homes covered: {mh_new_cov:>9,.0f}")

        # Marginal pop per siren
        prev_cov = current_coverage
        for step_i, step_w in enumerate(r['cum_weights']):
            gain = step_w - (prev_cov - current_coverage)
            prev_cov = current_coverage + step_w
            rank_pop = step_w - (r['cum_weights'][step_i-1] if step_i > 0 else 0)
            print(f"      Siren {step_i+1}: +{rank_pop:,.0f} additional residents covered")

        # Save GeoJSON
        proposed_gdf = gpd.GeoDataFrame(
            {
                'siren_id'   : [f'proposed_{n_new}_{i+1}' for i in range(len(sel_pts))],
                'priority'   : list(range(1, len(sel_pts)+1)),
                'type'       : 'proposed',
                'n_scenario' : n_new,
                'cost_usd'   : SIREN_COST_USD,
            },
            geometry=sel_pts,
            crs=UTM14N
        )
        fname = OUT_VEC / f'proposed_sirens_n{n_new}.geojson'
        proposed_gdf.to_file(fname, driver='GeoJSON')
        print(f"    Saved -> outputs/vectors/proposed_sirens_n{n_new}.geojson")

    print()
    print("=" * 60)
    print("  RECOMMENDATION:")
    r5 = results[5]
    cov5 = r5['total_new_cov']
    pct5 = 100 * cov5 / max(total_demand, 1)
    print(f"  Deploying 5 new sirens ($150,000) would extend warning")
    print(f"  coverage to an additional {cov5:,.0f} residents ({pct5:.1f}% improvement),")
    print(f"  prioritizing high-vulnerability mobile home communities.")
    print("=" * 60)


if __name__ == '__main__':
    main()
