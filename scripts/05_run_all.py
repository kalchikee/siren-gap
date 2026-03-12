"""
05_run_all.py
The Siren Gap - Master Pipeline Runner

Runs all analysis scripts in sequence with timing and progress output.
"""

import time
import importlib.util
import traceback
from pathlib import Path

ROOT    = Path(__file__).parent.parent
SCRIPTS = Path(__file__).parent


def run_script(name: str, filepath: Path) -> float:
    """
    Execute a script file and return elapsed seconds.

    Strategy:
      1. exec_module() runs module-level code (covers scripts 01 and 03).
      2. If the module exposes a main() function, call it explicitly
         (covers scripts 02 and 04 which guard their logic behind main()).
    """
    print()
    print("=" * 62)
    print(f"  RUNNING: {name}")
    print("=" * 62)
    t0 = time.perf_counter()

    spec   = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        # Call main() if the script uses that pattern and hasn't run yet
        if hasattr(module, 'main') and callable(module.main):
            module.main()
    except SystemExit:
        pass   # scripts that call sys.exit(0) are still OK

    elapsed = time.perf_counter() - t0
    print(f"\n  Completed in {elapsed:.1f}s")
    return elapsed


def main():
    pipeline_start = time.perf_counter()

    banner = """
+----------------------------------------------------------+
|         THE SIREN GAP - FULL ANALYSIS PIPELINE           |
|   Mapping Tornado Warning Dead Zones in Oklahoma          |
+----------------------------------------------------------+
"""
    print(banner)

    steps = [
        ("01_preprocess_data",     SCRIPTS / "01_preprocess_data.py"),
        ("02_acoustic_model",      SCRIPTS / "02_acoustic_model.py"),
        ("03_vulnerability_analysis", SCRIPTS / "03_vulnerability_analysis.py"),
        ("04_siren_optimization",  SCRIPTS / "04_siren_optimization.py"),
    ]

    timings = {}
    failed  = []

    for name, path in steps:
        if not path.exists():
            print(f"\n  ERROR: Script not found: {path}")
            failed.append(name)
            continue
        try:
            elapsed = run_script(name, path)
            timings[name] = elapsed
        except Exception as e:
            print(f"\n  ERROR in {name}:")
            traceback.print_exc()
            failed.append(name)
            timings[name] = -1.0

    # Summary
    total_elapsed = time.perf_counter() - pipeline_start
    print()
    print("=" * 62)
    print("  PIPELINE SUMMARY")
    print("=" * 62)
    for name, elapsed in timings.items():
        status = "FAILED" if elapsed < 0 else f"{elapsed:>6.1f}s"
        print(f"  {name:<40} {status}")
    print("-" * 62)
    print(f"  Total wall time: {total_elapsed:.1f}s")

    if failed:
        print(f"\n  FAILED scripts: {', '.join(failed)}")
    else:
        print("""
  All steps completed successfully.

  Key outputs:
    outputs/rasters/spl_composite.tif        - SPL acoustic model
    outputs/vectors/dead_zones.geojson        - warning dead zones
    outputs/vectors/vulnerability_scores.geojson - SGVS per block group
    outputs/vectors/proposed_sirens_n5.geojson   - optimal 5 new sirens
    web/index.html                            - interactive map
""")
    print("=" * 62)


if __name__ == '__main__':
    main()
