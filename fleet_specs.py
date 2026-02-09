"""
Fleet specs for Amazon delivery drones (Standard, Oversize, H&B).
Run calculate_fleet_specs(range_km) to print specs and write drone_fleet_specs.json.
L0 cell size (used across the project) = 1.5 * max(length, width) of the H&B drone.
"""

import json
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


def _l0_from_hb_dims(dims_str):
    """L0 cell size (m) = 1.5 * max(length, width) of H&B drone."""
    L, W, _ = _parse_dims_m(dims_str)
    return 1.5 * max(L, W)


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

    # L0 = 1.5 * max(length, width) of H&B
    hb_dims = FLEET_DATA["H&B"]["dims_LWH_m"]
    L0 = _l0_from_hb_dims(hb_dims)
    out["L0_cell_m"] = round(L0, 6)
    print(f"L0 cell size (1.5 × max H&B L/W): {L0:.4f} m")
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


def get_l0_cell_m(path=None, fallback_m=10.0):
    """
    Return L0 cell size in metres (1.5 × max H&B length/width).
    Reads from drone_fleet_specs.json if present; otherwise returns fallback_m.
    """
    specs = load_fleet_specs(path)
    if specs is not None and "L0_cell_m" in specs:
        return float(specs["L0_cell_m"])
    return fallback_m


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
