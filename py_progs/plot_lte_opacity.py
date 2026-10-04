#!/usr/bin/env python
"""
Plot an LTE opacity diagnostic from SIROCCO atomic data.

The script combines:
  - bound-free opacity from all available photoionization levels,
  - free-free opacity using the included Sutherland gaunt-factor table,
  - line opacity using thermally broadened Gaussian profiles.

Level populations are computed in LTE from the SIROCCO atomic data, so the
macro-atom levels already in the dataset are included explicitly.

Usage:

    plot_lte_opacity.py MASTERFILE TEMPERATURE NH [options]

Examples:

    plot_lte_opacity.py zdata/h20.dat 11600 6e9
    plot_lte_opacity.py data/standard80.dat 20000 1e10 --outfile lte_opacity.png

Options:

    --scan-dir DIR  Recursively scan this directory for atomic data files
                   instead of using the files listed in MASTERFILE.
    --outfile FILE  Output plot filename (default: auto-generated PNG)
    --tops-file FILE  Optional TOPS opacity table to overlay for comparison
    --table FILE    Optional ASCII table of the opacity curve
    --npoints N     Number of grid points in the base opacity grid (default: 1200)

"""

from __future__ import annotations

import argparse
import math
import os
import sys
from bisect import bisect_left
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ATOMIC_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "xdata", "atomic"))

EV2ERGS = 1.602192e-12
HPLANCK = 6.62607015e-27
CLIGHT = 2.99792458e10
KBOLTZ = 1.38062e-16
MELECTRON = 9.10938356e-28
AMU = 1.66053906660e-24
PI = math.pi
SQRT_PI = math.sqrt(math.pi)
RYD2ERGS = 2.1798723611035e-11
BREMS_CONSTANT = 3.692e8
ECHARGE = 4.803204712570263e-10


@dataclass
class PhotoRecord:
    source: str
    record_type: str
    z: int
    istate: int
    threshold_ev: float
    lower_level_id: object
    energies_ev: object
    sigma_cm2: object


@dataclass
class LineRecord:
    source: str
    record_type: str
    z: int
    istate: int
    wavelength_a: float
    oscillator_strength: float
    g_lower: float
    g_upper: float
    e_lower_ev: float
    e_upper_ev: float
    level_lower: int
    level_upper: int


@dataclass
class GauntRecord:
    log_gsqrd: float
    gff: float
    s1: float
    s2: float
    s3: float


def _import_numpy():
    import numpy as np

    return np


def _import_lte():
    return None


def resolve_source_path(path: str) -> str:
    if os.path.exists(path):
        return path

    if path.startswith("data/"):
        repo_root = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
        candidate = os.path.join(repo_root, path.replace("data/", "xdata/", 1))
        if os.path.exists(candidate):
            return candidate

    return path


def collect_ion_kinds(source_files: List[str]) -> Dict[Tuple[int, int], str]:
    ion_kinds: Dict[Tuple[int, int], str] = {}
    for path in source_files:
        with open(path, "r") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue
                label = parts[0]
                if label not in ("IonM", "IonV"):
                    continue
                key = (int(parts[2]), int(parts[3]))
                current = ion_kinds.get(key)
                if current is None or (current == "IonV" and label == "IonM"):
                    ion_kinds[key] = label
    return ion_kinds


def photo_record_matches_kind(record: PhotoRecord, ion_kind: Optional[str]) -> bool:
    if ion_kind == "IonM":
        return record.record_type == "PhotMac"
    if ion_kind == "IonV":
        return record.record_type in ("PhotTop", "PhotVfky")
    return True


def line_record_matches_kind(record: LineRecord, ion_kind: Optional[str]) -> bool:
    if ion_kind == "IonM":
        return record.record_type == "LinMacro"
    if ion_kind == "IonV":
        return record.record_type in ("CSTREN", "Line")
    return True


def parse_photo_file(path: str) -> List[PhotoRecord]:
    np = _import_numpy()
    records: List[PhotoRecord] = []
    current = None

    with open(path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            label = parts[0]

            if label in ("PhotTopS", "PhotVfkyS", "PhotMacS"):
                if current is not None:
                    records.append(
                        PhotoRecord(
                            source=path,
                            record_type=current["record_type"],
                            z=current["z"],
                            istate=current["istate"],
                            threshold_ev=current["threshold_ev"],
                            lower_level_id=current.get("lower_level_id"),
                            energies_ev=np.asarray(current["energies_ev"], dtype=float),
                            sigma_cm2=np.asarray(current["sigma_cm2"], dtype=float),
                        )
                    )

                lower_level_id = None
                if label == "PhotMacS" and len(parts) >= 4:
                    lower_level_id = int(parts[3])
                elif label in ("PhotTopS", "PhotVfkyS") and len(parts) >= 5:
                    lower_level_id = (int(parts[3]), int(parts[4]))

                current = {
                    "record_type": label[:-1],
                    "z": int(parts[1]),
                    "istate": int(parts[2]),
                    "threshold_ev": float(parts[5]),
                    "lower_level_id": lower_level_id,
                    "energies_ev": [],
                    "sigma_cm2": [],
                }
                continue

            if label in ("PhotTop", "PhotVfky", "PhotMac") and current is not None:
                current["energies_ev"].append(float(parts[1]))
                current["sigma_cm2"].append(float(parts[2]))

    if current is not None:
        records.append(
            PhotoRecord(
                source=path,
                record_type=current["record_type"],
                z=current["z"],
                istate=current["istate"],
                threshold_ev=current["threshold_ev"],
                lower_level_id=current.get("lower_level_id"),
                energies_ev=np.asarray(current["energies_ev"], dtype=float),
                sigma_cm2=np.asarray(current["sigma_cm2"], dtype=float),
            )
        )

    return records


def parse_tops_file(path: str):
    np = _import_numpy()

    energies_ev = []
    absorp_cm2_g = []
    reading_table = False
    with open(path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("T (keV), Density (gm/cc) ="):
                reading_table = True
                continue
            if not reading_table:
                continue
            if line.startswith("Photon energy(keV)"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                energy_keV = float(parts[0])
                absorption = float(parts[2])
            except ValueError:
                continue
            energies_ev.append(energy_keV * 1000.0)
            absorp_cm2_g.append(absorption)

    if not energies_ev:
        return None, None

    return np.asarray(energies_ev, dtype=float), np.asarray(absorp_cm2_g, dtype=float)


def parse_line_file(path: str) -> List[LineRecord]:
    records: List[LineRecord] = []
    seen = set()

    with open(path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if parts[0] not in ("CSTREN", "LinMacro", "Line") or len(parts) < 10:
                continue

            if parts[0] == "CSTREN":
                z = int(parts[2])
                istate = int(parts[3])
                wavelength_a = float(parts[4])
                oscillator_strength = float(parts[5])
                g_lower = float(parts[6])
                g_upper = float(parts[7])
                e_lower_ev = float(parts[8])
                e_upper_ev = float(parts[9])
                level_lower = int(parts[10])
                level_upper = int(parts[11])
            else:
                z = int(parts[1])
                istate = int(parts[2])
                wavelength_a = float(parts[3])
                oscillator_strength = float(parts[4])
                g_lower = float(parts[5])
                g_upper = float(parts[6])
                e_lower_ev = float(parts[7])
                e_upper_ev = float(parts[8])
                level_lower = int(parts[9])
                level_upper = int(parts[10])

            key = (
                z,
                istate,
                round(wavelength_a, 8),
                round(oscillator_strength, 12),
                round(g_lower, 8),
                round(g_upper, 8),
                round(e_lower_ev, 8),
                round(e_upper_ev, 8),
                level_lower,
                level_upper,
            )
            if key in seen:
                continue
            seen.add(key)

            records.append(
                LineRecord(
                    source=path,
                    record_type=parts[0],
                    z=z,
                    istate=istate,
                    wavelength_a=wavelength_a,
                    oscillator_strength=oscillator_strength,
                    g_lower=g_lower,
                    g_upper=g_upper,
                    e_lower_ev=e_lower_ev,
                    e_upper_ev=e_upper_ev,
                    level_lower=level_lower,
                    level_upper=level_upper,
                )
            )

    return records


def parse_gaunt_file(path: str) -> List[GauntRecord]:
    records: List[GauntRecord] = []
    seen = set()

    with open(path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if parts[0] != "FF_GAUNT" or len(parts) < 6:
                continue

            log_gsqrd = float(parts[1])
            key = round(log_gsqrd, 8)
            if key in seen:
                continue
            seen.add(key)

            records.append(
                GauntRecord(
                    log_gsqrd=log_gsqrd,
                    gff=float(parts[2]),
                    s1=float(parts[3]),
                    s2=float(parts[4]),
                    s3=float(parts[5]),
                )
            )

    records.sort(key=lambda item: item.log_gsqrd)
    return records


def discover_source_files(masterfile: str, scan_dir: str = "") -> List[str]:
    if scan_dir:
        discovered = []
        for root, _, files in os.walk(scan_dir):
            for name in files:
                if name.lower().endswith(".dat"):
                    discovered.append(os.path.join(root, name))
        return sorted(discovered)

    return [resolve_source_path(path) for path in parse_masterfile(masterfile)]


def parse_masterfile(masterfile: str) -> List[str]:
    data_files: List[str] = []
    master_dir = os.path.dirname(os.path.abspath(masterfile))
    with open(masterfile, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            candidate = resolve_source_path(line)
            if os.path.exists(candidate):
                data_files.append(candidate)
                continue
            relative_candidate = os.path.join(master_dir, line)
            relative_candidate = resolve_source_path(relative_candidate)
            if os.path.exists(relative_candidate):
                data_files.append(relative_candidate)
    return data_files


def load_atomic_records(masterfile: str, scan_dir: str = ""):
    source_files = discover_source_files(masterfile, scan_dir)
    ion_kinds = collect_ion_kinds(source_files)
    photo_records: List[PhotoRecord] = []
    line_records: List[LineRecord] = []
    gaunt_records: List[GauntRecord] = []

    for path in source_files:
        path = resolve_source_path(path)
        if not os.path.exists(path):
            continue

        with open(path, "r") as handle:
            text = handle.read()

        if "PhotTop" in text or "PhotVfky" in text or "PhotMac" in text:
            for record in parse_photo_file(path):
                if photo_record_matches_kind(record, ion_kinds.get((record.z, record.istate))):
                    photo_records.append(record)
        if "CSTREN" in text or "LinMacro" in text or "Line" in text:
            for record in parse_line_file(path):
                if line_record_matches_kind(record, ion_kinds.get((record.z, record.istate))):
                    line_records.append(record)
        if "FF_GAUNT" in text:
            gaunt_records.extend(parse_gaunt_file(path))

    return photo_records, line_records, gaunt_records


def build_lte_state(masterfile: str, temperature: float, nh: float):
    np = _import_numpy()

    elements, ions, levels = parse_atomic_data(masterfile)
    level_lookup: Dict[Tuple[int, int], List[Tuple[int, object]]] = {}
    level_id_lookup: Dict[Tuple[int, int, int], int] = {}
    for index, level in enumerate(levels):
        level_lookup.setdefault((level["z"], level["istate"]), []).append((index, level))
        if level.get("level_id") is not None:
            level_id_lookup[(level["z"], level["istate"], level["level_id"])] = index

    ion_partition: Dict[Tuple[int, int], float] = {}
    ion_density: Dict[Tuple[int, int], float] = {}
    level_populations: Dict[Tuple[int, int, int], float] = {}
    ion_results: List[Dict] = []

    for ion in ions:
        key = (ion["z"], ion["istate"])
        if ion["z"] not in elements:
            ion_partition[key] = ion["g"]
            continue
        partition = 0.0
        for _, level in level_lookup.get(key, []):
            partition += level["g"] * np.exp(-level["ex_ev"] * EV2ERGS / (KBOLTZ * temperature))
        ion_partition[key] = max(partition, ion["g"])

    h1_key = (1, 1)
    xsaha = 4.82907e15 * temperature ** 1.5
    if h1_key in { (ion["z"], ion["istate"]) for ion in ions }:
        h1 = next(ion for ion in ions if ion["z"] == 1 and ion["istate"] == 1)
        theta = xsaha * math.exp(-h1["ip_ev"] * EV2ERGS / (KBOLTZ * temperature)) / nh
        if theta < 1.0e4:
            x = (-theta + math.sqrt(theta * theta + 4.0 * theta)) / 2.0
            ne = x * nh
        else:
            ne = nh
    else:
        ne = 0.5 * nh

    ne = max(ne, 1.0e-6)
    for _ in range(200):
        densities: Dict[Tuple[int, int], float] = {}
        by_element: Dict[int, List[Dict]] = {}
        for ion in ions:
            by_element.setdefault(ion["z"], []).append(ion)

        for z, ion_list in by_element.items():
            if z not in elements:
                for ion_stage in ion_list:
                    densities[(ion_stage["z"], ion_stage["istate"])] = 0.0
                continue
            ion_list = sorted(ion_list, key=lambda item: item["istate"])
            element = elements.get(z)
            if element is None:
                # Some masterfiles list ion stages for elements whose abundance lines are commented out.
                # Treat those elements as zero abundance instead of failing.
                for ion_stage in ion_list:
                    stage_key = (ion_stage["z"], ion_stage["istate"])
                    densities[stage_key] = 0.0
                continue

            base_density = nh * element["abun"]
            stage_relative = [1.0]
            for prev, curr in zip(ion_list[:-1], ion_list[1:]):
                prev_key = (prev["z"], prev["istate"])
                curr_key = (curr["z"], curr["istate"])
                ionization_potential_ev = prev["ip_ev"]
                ratio = (
                    xsaha
                    * ion_partition[curr_key]
                    * math.exp(-ionization_potential_ev * EV2ERGS / (KBOLTZ * temperature))
                    / (ne * ion_partition[prev_key])
                )
                stage_relative.append(stage_relative[-1] * ratio)

            total_relative = sum(stage_relative)
            if total_relative <= 0.0:
                total_relative = 1.0

            for idx, ion_stage in enumerate(ion_list):
                stage_key = (ion_stage["z"], ion_stage["istate"])
                densities[stage_key] = base_density * stage_relative[idx] / total_relative

        ne_new = 0.0
        for ion in ions:
            density = densities[(ion["z"], ion["istate"])]
            ne_new += density * max(ion["istate"] - 1, 0)

        if ne_new <= 0.0:
            ne_new = 1.0e-6
        if abs(ne - ne_new) / ne_new < 0.03:
            ne = ne_new
            ion_density = densities
            break
        ne = 0.5 * (ne + ne_new)
        ion_density = densities

    for ion in ions:
        key = (ion["z"], ion["istate"])
        density = ion_density.get(key, 0.0)
        level_list = level_lookup.get(key, [])
        total = 0.0
        for _, level in level_list:
            total += level["g"] * np.exp(-level["ex_ev"] * EV2ERGS / (KBOLTZ * temperature))
        total = max(total, ion["g"])

        for level_index, level in level_list:
            level_density = density * level["g"] * np.exp(-level["ex_ev"] * EV2ERGS / (KBOLTZ * temperature)) / total
            level_populations[(ion["z"], ion["istate"], level_index)] = level_density
            ion_results.append(
                {
                    "z": ion["z"],
                    "istate": ion["istate"],
                    "density": density,
                    "ne": ne,
                    "level_index": level_index,
                    "level_density": level_density,
                }
            )

    element_mass = {z: elem["atomic_weight"] * AMU for z, elem in elements.items()}
    mass_density = 0.0
    for elem in elements.values():
        mass_density += nh * elem["abun"] * elem["atomic_weight"] * AMU

    return ion_results, ion_density, ne, level_lookup, level_id_lookup, level_populations, element_mass, mass_density


def find_matching_excitation_level(
    z: int,
    istate: int,
    target_ev: float,
    level_lookup: Dict[Tuple[int, int], List[Tuple[int, object]]],
) -> Optional[int]:
    candidates = level_lookup.get((z, istate), [])
    if not candidates:
        return None

    best_index = None
    best_delta = None
    for level_index, level in candidates:
        delta = abs(level["ex_ev"] - target_ev)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_index = level_index
    return best_index


def find_matching_threshold_level(
    z: int,
    istate: int,
    target_ev: float,
    level_lookup: Dict[Tuple[int, int], List[Tuple[int, object]]],
) -> Optional[int]:
    candidates = level_lookup.get((z, istate), [])
    if not candidates:
        return None

    best_index = None
    best_delta = None
    for level_index, level in candidates:
        delta = abs(level["ion_pot_ev"] - target_ev)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_index = level_index
    return best_index


def parse_atomic_data(masterfile: str):
    elements: Dict[int, Dict] = {}
    ions: List[Dict] = []
    levels: List[Dict] = []
    source_files = discover_source_files(masterfile)
    ion_kinds = collect_ion_kinds(source_files)

    for path in source_files:
        with open(path, "r") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                label = parts[0]
                if label == "Element" and len(parts) >= 5:
                    z = int(parts[1])
                    elements[z] = {
                        "name": parts[2],
                        "abun_log": float(parts[3]),
                        "atomic_weight": float(parts[4]),
                    }
                elif label in ("IonM", "IonV") and len(parts) >= 6:
                    ions.append(
                        {
                            "z": int(parts[2]),
                            "istate": int(parts[3]),
                            "g": float(parts[4]),
                            "ip_ev": float(parts[5]),
                            "kind": label,
                        }
                    )
                elif label == "LevMacro" and len(parts) >= 7:
                    if ion_kinds.get((int(parts[1]), int(parts[2]))) not in (None, "IonM"):
                        continue
                    ex_ev = float(parts[5])
                    ion_pot_ev = abs(float(parts[4]))
                    levels.append(
                        {
                            "z": int(parts[1]),
                            "istate": int(parts[2]),
                            "level_id": int(parts[3]),
                            "g": float(parts[6]),
                            "ion_pot_ev": ion_pot_ev,
                            "ex_ev": ex_ev,
                            "ex": ex_ev * EV2ERGS,
                        }
                    )
                elif label == "LevTop" and len(parts) >= 8:
                    if ion_kinds.get((int(parts[1]), int(parts[2]))) not in (None, "IonV"):
                        continue
                    ex_ev = float(parts[6])
                    levels.append(
                        {
                            "z": int(parts[1]),
                            "istate": int(parts[2]),
                            "level_id": (int(parts[3]), int(parts[4])),
                            "g": float(parts[7]),
                            "ion_pot_ev": abs(float(parts[5])),
                            "ex_ev": ex_ev,
                            "ex": ex_ev * EV2ERGS,
                        }
                    )
                elif label == "Level" and len(parts) >= 6:
                    if ion_kinds.get((int(parts[1]), int(parts[2]))) not in (None, "IonV"):
                        continue
                    ex_ev = float(parts[5])
                    levels.append(
                        {
                            "z": int(parts[1]),
                            "istate": int(parts[2]),
                            "level_id": int(parts[3]),
                            "g": float(parts[4]),
                            "ion_pot_ev": float("nan"),
                            "ex_ev": ex_ev,
                            "ex": ex_ev * EV2ERGS,
                        }
                    )
    for elem in elements.values():
        elem["abun"] = 10 ** (elem["abun_log"] - 12.0)
    return elements, ions, levels


def interpolate_photo_xsection(record: PhotoRecord, energy_grid_ev):
    np = _import_numpy()

    energies = np.asarray(record.energies_ev, dtype=float)
    sigma = np.asarray(record.sigma_cm2, dtype=float)

    valid = np.isfinite(energies) & np.isfinite(sigma) & (energies > 0.0) & (sigma > 0.0)
    energies = energies[valid]
    sigma = sigma[valid]

    if len(energies) == 0:
        return np.zeros_like(energy_grid_ev)

    order = np.argsort(energies)
    energies = energies[order]
    sigma = sigma[order]

    result = np.zeros_like(energy_grid_ev)
    mask = energy_grid_ev >= energies[0]
    if np.any(mask):
        result[mask] = np.exp(np.interp(np.log(energy_grid_ev[mask]), np.log(energies), np.log(sigma)))
    return result


def gaunt_ff(records: List[GauntRecord], gsqrd: float) -> float:
    if not records or gsqrd <= 0.0:
        return 1.0

    log_g2 = math.log10(gsqrd)
    if log_g2 < records[0].log_gsqrd or log_g2 > records[-1].log_gsqrd:
        return 1.0

    logs = [item.log_gsqrd for item in records]
    index = bisect_left(logs, log_g2)
    if index <= 0:
        index = 1
    if index >= len(records):
        index = len(records) - 1

    lower = records[index - 1]
    delta = log_g2 - lower.log_gsqrd
    return lower.gff + delta * (lower.s1 + delta * (lower.s2 + delta * lower.s3))


def build_energy_grid(photo_records: List[PhotoRecord], line_records: List[LineRecord], npoints: int):
    np = _import_numpy()

    thresholds = [record.threshold_ev for record in photo_records if record.threshold_ev > 0.0]
    photo_high = []
    for record in photo_records:
        energies = getattr(record, "energies_ev", None)
        if energies is not None and len(energies) > 0:
            photo_high.append(float(np.max(energies)))

    line_centers = []
    for record in line_records:
        energy0 = (HPLANCK * CLIGHT / (record.wavelength_a * 1.0e-8)) / EV2ERGS
        if energy0 > 0.0:
            line_centers.append(energy0)

    if thresholds or line_centers:
        lower = min(thresholds + line_centers)
    else:
        lower = 1.0

    if photo_high or line_centers:
        upper = max(photo_high + line_centers)
    else:
        upper = lower * 10.0

    if upper <= lower:
        upper = lower * 10.0

    lower = max(lower * 0.999, 1.0e-4)
    base = np.logspace(np.log10(lower), np.log10(upper), npoints)
    inserts = [base]
    if line_centers:
        inserts.append(np.asarray(line_centers, dtype=float))
    if thresholds:
        inserts.append(np.asarray(thresholds, dtype=float))
    return np.unique(np.concatenate(inserts))


def compute_continuum_components(
    energy_grid_ev,
    temperature: float,
    ion_results: List[Dict],
    ion_density: Dict[Tuple[int, int], float],
    photo_records: List[PhotoRecord],
    level_lookup: Dict[Tuple[int, int], List[Tuple[int, object]]],
    level_id_lookup: Dict[Tuple[int, int, int], int],
    level_populations: Dict[Tuple[int, int, int], float],
    gaunt_records: List[GauntRecord],
    electron_density: float = None,
):
    np = _import_numpy()

    kappa_bf = np.zeros_like(energy_grid_ev)
    kappa_ff = np.zeros_like(energy_grid_ev)

    for record in photo_records:
        level_index = None
        if record.lower_level_id is not None:
            level_index = level_id_lookup.get((record.z, record.istate, record.lower_level_id))
        if level_index is None:
            level_index = find_matching_threshold_level(record.z, record.istate, record.threshold_ev, level_lookup)
        if level_index is None:
            density = ion_density.get((record.z, record.istate), 0.0)
        else:
            density = level_populations.get((record.z, record.istate, level_index), 0.0)

        if density <= 0.0:
            continue

        sigma = interpolate_photo_xsection(record, energy_grid_ev)
        kappa_bf += density * sigma

    kT_erg = KBOLTZ * temperature
    nu_grid = energy_grid_ev * EV2ERGS / HPLANCK
    stim = 1.0 - np.exp(-HPLANCK * nu_grid / kT_erg)

    ne = 0.0
    free_free_sum = np.zeros_like(energy_grid_ev)
    for (z, istate), density in ion_density.items():
        charge = max(istate - 1, 0)
        if charge <= 0:
            continue
        gsqrd = (charge * charge * RYD2ERGS) / kT_erg
        gaunt = gaunt_ff(gaunt_records, gsqrd)
        ne += density * charge
        free_free_sum += density * charge * charge * gaunt

    if electron_density is not None:
        if electron_density <= 0.0:
            raise ValueError("electron density must be positive")
        ne = electron_density

    if ne > 0.0:
        kappa_ff = BREMS_CONSTANT * ne * free_free_sum * stim / (np.sqrt(temperature) * nu_grid ** 3)

    return kappa_bf, kappa_ff


def compute_line_component(
    energy_grid_ev,
    temperature: float,
    ion_density: Dict[Tuple[int, int], float],
    level_lookup: Dict[Tuple[int, int], List[Tuple[int, object]]],
    level_populations: Dict[Tuple[int, int, int], float],
    line_records: List[LineRecord],
    element_mass: Dict[int, float],
):
    np = _import_numpy()

    kappa_line = np.zeros_like(energy_grid_ev)
    kT_erg = KBOLTZ * temperature
    line_prefactor = PI * (ECHARGE ** 2) / (MELECTRON * CLIGHT) * (HPLANCK / EV2ERGS)

    for record in line_records:
        ion_key = (record.z, record.istate)
        if ion_key not in ion_density:
            continue

        mass = element_mass.get(record.z)
        if mass is None or mass <= 0.0:
            continue

        level_index = find_matching_excitation_level(record.z, record.istate, record.e_lower_ev, level_lookup)
        if level_index is None:
            lower_density = ion_density[ion_key]
        else:
            lower_density = level_populations.get((record.z, record.istate, level_index), ion_density[ion_key])

        if lower_density <= 0.0 or record.oscillator_strength <= 0.0:
            continue

        energy0_ev = (HPLANCK * CLIGHT / (record.wavelength_a * 1.0e-8)) / EV2ERGS
        if energy0_ev <= 0.0:
            continue

        thermal_velocity = math.sqrt(2.0 * KBOLTZ * temperature / mass)
        sigma_energy = energy0_ev * thermal_velocity / CLIGHT
        if sigma_energy <= 0.0:
            continue

        profile = np.exp(-((energy_grid_ev - energy0_ev) / sigma_energy) ** 2) / (sigma_energy * SQRT_PI)
        stim = 1.0 - np.exp(-energy0_ev * EV2ERGS / kT_erg)
        kappa_line += line_prefactor * lower_density * record.oscillator_strength * stim * profile

    return kappa_line


def write_table(filename: str, energy_ev, kappa_bf, kappa_ff, kappa_line):
    total = kappa_bf + kappa_ff + kappa_line
    with open(filename, "w") as handle:
        handle.write("# energy_ev bound_free_cm2_per_g free_free_cm2_per_g line_cm2_per_g total_cm2_per_g\n")
        for e, bf, ff, line, tot in zip(energy_ev, kappa_bf, kappa_ff, kappa_line, total):
            handle.write(f"{e:15.8e} {bf:15.8e} {ff:15.8e} {line:15.8e} {tot:15.8e}\n")


def plot_opacity(
    energy_ev,
    kappa_bf,
    kappa_ff,
    kappa_line,
    tops_energy_ev,
    tops_absorp_cm2_g,
    outfile: str,
    masterfile: str,
    temperature: float,
    nh: float,
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to create the opacity plot") from exc

    total = kappa_bf + kappa_ff + kappa_line

    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    ax.loglog(energy_ev, total, color="black", linewidth=2.2, label="total")
    ax.loglog(energy_ev, kappa_bf, color="#1f77b4", linewidth=1.5, label="bound-free")
    ax.loglog(energy_ev, kappa_ff, color="#d62728", linewidth=1.3, label="free-free")
    ax.loglog(energy_ev, kappa_line, color="#2ca02c", linewidth=1.0, alpha=0.3, label="lines")
    if tops_energy_ev is not None and tops_absorp_cm2_g is not None:
        ax.loglog(
            tops_energy_ev,
            tops_absorp_cm2_g,
            color="#9467bd",
            linewidth=1.4,
            linestyle="--",
            label="TOPS absorption", alpha=0.7,
        )

    ax.set_xlabel("Photon energy (eV)")
    ax.set_ylabel("Opacity (cm$^2$ g$^{-1}$)")
    ax.set_xlim(left=max(energy_ev[0], 1.0e-4), right=energy_ev[-1] * 1.01)
    ax.set_xlim(1,15*KBOLTZ*temperature/EV2ERGS)
    positive = total[total > 0.0]
    if positive.size > 0:
        ymin = max(float(positive.min()) * 0.5, 1.0e-30)
        ymax = float(positive.max()) * 2.0
        if ymax > ymin:
            ax.set_ylim(bottom=ymin, top=ymax)
    ax.set_ylim([1.e-4,1.e3])
    ax.tick_params(axis="both", which="major", length=6, width=1.2)
    ax.tick_params(axis="both", which="minor", length=4, width=1.0)
    ax.minorticks_on()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(
        f"LTE opacity from {os.path.basename(masterfile)}\n"
        f"T = {temperature:.3g} K, nh = {nh:.3g} cm$^{{-3}}$"
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(outfile, dpi=180)


def parse_args(argv: List[str]):
    parser = argparse.ArgumentParser(description="Plot an LTE opacity diagnostic from SIROCCO atomic data.")
    parser.add_argument("masterfile", help="Atomic data masterfile")
    parser.add_argument("temperature", type=float, help="LTE temperature in K")
    parser.add_argument("nh", type=float, help="Hydrogen number density in cm^-3")
    parser.add_argument(
        "--scan-dir",
        default="",
        help="Optional directory to scan recursively instead of using the masterfile references",
    )
    parser.add_argument(
        "--outfile",
        default="",
        help="Output plot filename (default: auto-generated PNG)",
    )
    parser.add_argument(
        "--tops-file",
        default="",
        help="Optional TOPS opacity table to overlay for comparison",
    )
    parser.add_argument(
        "--table",
        default="",
        help="Optional ASCII table filename for the opacity curve",
    )
    parser.add_argument(
        "--npoints",
        type=int,
        default=1200,
        help="Number of grid points in the base opacity grid",
    )
    return parser.parse_args(argv)


def main(argv: List[str]):
    args = parse_args(argv)
    print('calculating LTE opacity for:')
    print(f"  Masterfile: {args.masterfile}")
    print(f"  Temperature: {args.temperature} K")
    print(f"  Hydrogen density: {args.nh} cm^-3")

    outfile = args.outfile
    if not outfile:
        root = os.path.splitext(os.path.basename(args.masterfile))[0]
        outfile = f"{root}_lte_opacity.png"

    photo_records, line_records, gaunt_records = load_atomic_records(args.masterfile, args.scan_dir)
    if not photo_records:
        raise RuntimeError("No photoionization records were found in the selected atomic data")

    ion_results, ion_density, ne, level_lookup, level_id_lookup, level_populations, element_mass, mass_density = build_lte_state(
        args.masterfile,
        args.temperature,
        args.nh,
    )

    energy_grid_ev = build_energy_grid(photo_records, line_records, args.npoints)
    kappa_bf, kappa_ff = compute_continuum_components(
        energy_grid_ev,
        args.temperature,
        ion_results,
        ion_density,
        photo_records,
        level_lookup,
        level_id_lookup,
        level_populations,
        gaunt_records,
    )
    kappa_line = compute_line_component(
        energy_grid_ev,
        args.temperature,
        ion_density,
        level_lookup,
        level_populations,
        line_records,
        element_mass,
    )

    tops_energy_ev = None
    tops_absorp_cm2_g = None
    if args.tops_file:
        tops_energy_ev, tops_absorp_cm2_g = parse_tops_file(args.tops_file)

    if mass_density <= 0.0:
        raise RuntimeError("Failed to compute a positive gas mass density for opacity normalization")
    print(f"Computed mass density: {mass_density:.3e} g cm^-3")
    kappa_bf = kappa_bf / mass_density
    kappa_ff = kappa_ff / mass_density
    kappa_line = kappa_line / mass_density

    if args.table:
        write_table(args.table, energy_grid_ev, kappa_bf, kappa_ff, kappa_line)

    plot_opacity(
        energy_grid_ev,
        kappa_bf,
        kappa_ff,
        kappa_line,
        tops_energy_ev,
        tops_absorp_cm2_g,
        outfile,
        args.masterfile,
        args.temperature,
        args.nh,
    )

    print(f"Wrote {outfile}")
    if args.table:
        print(f"Wrote {args.table}")


if __name__ == "__main__":
    main(sys.argv[1:])
