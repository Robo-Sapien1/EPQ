"""
Fleet specs for Amazon delivery drones (Standard, Oversize, H&B).
Run calculate_fleet_specs(range_km) to print specs and write drone_fleet_specs.json.

Spatial units derived from the H&B drone (largest tier):
  atomic_unit_m  = max(length, width) + 2 * lateral_padding_m
                 = AirMatrix cell side: largest drone footprint + padding
  L0_cell_m      = 4 * atomic_unit_m
                 = the bottom layer cell size (a 4x4 "bucket" of atomic units)
"""

import json
import math
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent
FLEET_SPECS_JSON = OUTPUT_DIR / "drone_fleet_specs.json"

# --- FLEET DATA (enclosure size and payload per tier) ---
FLEET_DATA = {
    "Standard": {
        "payload_kg": 11.9,
        "dims_LWH_m": "1.0 x 1.0 x 0.6",
        "desc": "Medium/Large Parcels",
    },
    "Oversize": {
        "payload_kg": 29.8,
        "dims_LWH_m": "1.8 x 1.5 x 1.0",
        "desc": "Standard Oversize (Microwaves, Monitors)",
    },
    "H&B": {
        "payload_kg": 45.0,
        "dims_LWH_m": "2.5 x 2.0 x 1.2",
        "desc": "Heavy & Bulky (Furniture, Gym Equipment)",
    },
}


def _parse_dims_m(dims_str):
    """Parse 'L x W x H' string; return (length_m, width_m, height_m)."""
    parts = [float(x.strip()) for x in dims_str.replace("x", " ").split()]
    if len(parts) != 3:
        raise ValueError(f"Expected 'L x W x H', got: {dims_str!r}")
    return parts[0], parts[1], parts[2]


def _atomic_unit_from_hb_dims(dims_str):
    """
    Atomic unit (m) for the system's finest *discrete occupancy* space (AirMatrix).

    We model the AirMatrix as a square/cube lattice where a drone occupies one
    cell at a time near delivery. If two drones occupy adjacent cells, their
    commanded centers are one cell-size apart; to prevent overlap we add a
    lateral padding margin to the drone footprint.

    Definition:
      atomic_unit = max(L, W) + 2 * lateral_padding_m

    Where lateral_padding_m represents "how far the drone can deviate from the
    cell centre" in terminal operations. A defensible default is ~0.3 m, matching
    typical vision-based hovering accuracy order-of-magnitude for capable
    multirotors (see DRONE_DIMENSIONS_FUNCTION_EXPLAINED.md for justification).
    """
    L, W, _H = _parse_dims_m(dims_str)
    lateral_padding_m = 0.30
    return float(max(L, W) + 2.0 * lateral_padding_m)


# L0 bucket factor: L0 cell = 4 * atomic_unit (a 4x4 cluster of atomic units)
L0_BUCKET_FACTOR = 4


def calculate_fleet_specs(range_km, output_path=None):
    """
    Takes a range (km) and calculates the required drone size and weight
    for all 3 Amazon delivery classes based on 2035 technology.
    Prints a summary and writes dimensions (including L0_cell_m) to a JSON file.
    """
    output_path = output_path or FLEET_SPECS_JSON

    print("\n" + "=" * 60)
    print("   FUTURE DRONE FLEET MODELLING (2035 ERA)")
    print(f"   Target Operational Range: {range_km} km")
    print("=" * 60 + "\n")

    # Physics constants (2035 assumptions)
    structural_fraction = 0.30
    k_factor = 0.0006
    available_mass_fraction = 1.0 - structural_fraction

    out = {
        "range_km": range_km,
        "drones": {},
        "atomic_unit_m": None,
        "L0_cell_m": None,
    }

    for drone_type, specs in FLEET_DATA.items():
        w_payload = specs["payload_kg"]
        dims_str = specs["dims_LWH_m"]
        denominator = available_mass_fraction - (range_km * k_factor)

        print(f"[{drone_type.upper()} DRONE] ({specs['desc']})")
        print(f"   Enclosure Size Needed: {dims_str} meters")
        print(f"   Max Payload Capacity:  {w_payload} kg")

        drone_out = {
            "payload_kg": w_payload,
            "dims_LWH_m": dims_str,
            "desc": specs["desc"],
        }

        if denominator <= 0:
            print("   Status: FAILED (Range too far for this tech level)")
            drone_out["status"] = "FAILED"
            drone_out["total_kg"] = None
            drone_out["battery_kg"] = None
            drone_out["frame_motor_kg"] = None
        else:
            w_total = w_payload / denominator
            w_struct = w_total * structural_fraction
            w_batt = w_total - w_struct - w_payload
            print(f"   > Total Takeoff Weight: {w_total:.2f} kg")
            print(f"   > Battery Weight:       {w_batt:.2f} kg")
            print(f"   > Frame/Motor Weight:   {w_struct:.2f} kg")
            drone_out["status"] = "OK"
            drone_out["total_kg"] = round(w_total, 2)
            drone_out["battery_kg"] = round(w_batt, 2)
            drone_out["frame_motor_kg"] = round(w_struct, 2)

        out["drones"][drone_type] = drone_out
        print("-" * 40)

    # Atomic unit (AirMatrix cell side) = max(length, width) + 2*padding of H&B
    # L0 cell     = 4 * atomic unit
    hb_dims = FLEET_DATA["H&B"]["dims_LWH_m"]
    atomic = _atomic_unit_from_hb_dims(hb_dims)
    l0_cell = L0_BUCKET_FACTOR * atomic
    out["atomic_unit_m"] = round(atomic, 6)
    out["L0_cell_m"] = round(l0_cell, 6)
    print(f"Atomic unit (max H&B L/W + 2*padding)  : {atomic:.4f} m")
    print(f"L0 cell size ({L0_BUCKET_FACTOR} x atomic unit)  : {l0_cell:.4f} m")
    print()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Written: {output_path}")

    return out


def load_fleet_specs(path=None):
    """
    Load fleet specs from JSON. Returns dict with keys range_km, drones, L0_cell_m,
    or None if file is missing/invalid.
    """
    path = path or FLEET_SPECS_JSON
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def get_atomic_unit_m(path=None, fallback_m=3.1):
    """
    Return the atomic unit size in metres.

    This is the AirMatrix cell side for 1 drone:
        max(H&B length, width) + 2 * lateral_padding_m
    Reads from drone_fleet_specs.json if present; otherwise returns fallback_m.

    With current H&B dims (2.5 x 2.0 x 1.2) and padding=0.30 m:
        2.5 + 0.60 = 3.10 m
    """
    specs = load_fleet_specs(path)
    hb_dims = None
    if specs is not None and "drones" in specs and "H&B" in specs["drones"]:
        hb_dims = specs["drones"]["H&B"].get("dims_LWH_m")
    if not hb_dims:
        hb_dims = FLEET_DATA["H&B"]["dims_LWH_m"]

    computed = float(_atomic_unit_from_hb_dims(hb_dims))

    # If a JSON exists but contains an older atomic unit definition, prefer the
    # current computed value so the system updates automatically.
    if specs is not None and "atomic_unit_m" in specs:
        try:
            stored = float(specs["atomic_unit_m"])
            if math.isfinite(stored) and stored > 0 and abs(stored - computed) < 1e-6:
                return stored
        except Exception:
            pass
    return computed


def get_l0_cell_m(path=None, fallback_m=12.4):
    """
    Return the L0 (bottom layer) cell size in metres.

    L0 cell = 4 * atomic_unit.  This is the 4x4 "bucket" that serves as
    the finest resolution of the pathfinding grid (not atomic, but the
    smallest graph node).

    With current H&B dims and padding=0.30 m:  4 * 3.10 = 12.4 m.
    """
    specs = load_fleet_specs(path)
    computed = float(L0_BUCKET_FACTOR * get_atomic_unit_m(path))

    if specs is not None and "L0_cell_m" in specs:
        try:
            stored = float(specs["L0_cell_m"])
            if math.isfinite(stored) and stored > 0 and abs(stored - computed) < 1e-6:
                return stored
        except Exception:
            pass
    return computed


def get_drone_dims(drone_type, path=None):
    """
    Return (length_m, width_m, height_m) for a drone type (Standard, Oversize, H&B).
    Reads from drone_fleet_specs.json if present; otherwise uses FLEET_DATA.
    """
    specs = load_fleet_specs(path)
    if specs is not None and "drones" in specs and drone_type in specs["drones"]:
        dims_str = specs["drones"][drone_type].get("dims_LWH_m")
        if dims_str:
            return _parse_dims_m(dims_str)
    if drone_type in FLEET_DATA:
        return _parse_dims_m(FLEET_DATA[drone_type]["dims_LWH_m"])
    return (1.0, 1.0, 0.6)


# --- MAIN: run with your desired range to regenerate the specs file ---
if __name__ == "__main__":
    range_km = float(input("Target operational range (km): "))
    calculate_fleet_specs(range_km=range_km)
