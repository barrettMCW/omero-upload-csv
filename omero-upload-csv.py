#!/usr/bin/env python3
"""
Script to read a CSV file and assign key-value pairs to OMERO datasets or images.
- If row has subject_id: assign to OMERO dataset
- If row has sample_id: assign to OMERO image
- If both present: prioritize sample_id (assign to image)
"""

import argparse
import csv
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

from omero.gateway import BlitzGateway, DatasetWrapper, ImageWrapper, MapAnnotationWrapper


def connect_to_omero(host: str, port: int, username: str, password: str, secure: bool) -> BlitzGateway:
    """Connect to OMERO server with explicit parameters."""
    conn = BlitzGateway(username, password, host=host, port=port, secure=secure)
    if not conn.connect():
        raise ConnectionError(f"Failed to connect to OMERO at {host}:{port}")
    
    # optionally set a group if desired; left here to mirror previous behavior if needed
    try:
        conn.SERVICE_OPTS.setOmeroGroup(219)
    except Exception:
        # not critical; some gateways don't expose SERVICE_OPTS
        pass

    logging.info("Connected to OMERO server: %s:%s (secure=%s)", host, port, secure)
    return conn


MAP_ANNOTATION_NS = "openmicroscopy.org/omero/client/mapAnnotation"


def get_map_annotations(obj) -> List[MapAnnotationWrapper]:
    """Return all map annotations on the object matching our namespace.

    Use listAnnotations instead of getAnnotations because some BlitzGateway wrappers
    do not support ns filtering with getAnnotations consistently.
    """
    try:
        anns = list(obj.listAnnotations(ns=MAP_ANNOTATION_NS))
    except Exception:
        anns = []

    if not anns:
        # Fallback: fetch all annotations and filter ourselves by namespace
        try:
            anns = [ann for ann in obj.listAnnotations() if getattr(ann, "getNs", lambda: None)() == MAP_ANNOTATION_NS]
        except Exception:
            anns = []

    return [ann for ann in anns if isinstance(ann, MapAnnotationWrapper)]


def map_annotation_to_dict(annotation: Optional[MapAnnotationWrapper]) -> Dict[str, str]:
    """Convert an existing MapAnnotation to a dict of key-values."""
    if not annotation:
        return {}
    values: Dict[str, str] = {}
    for entry in annotation.getValue():
        if not isinstance(entry, tuple) or len(entry) < 2:
            continue
        key, value = entry[0], entry[1]
        values[str(key)] = str(value)
    return values


def consolidate_map_annotations(conn: BlitzGateway, obj, absolute: bool = False) -> Tuple[Optional[MapAnnotationWrapper], Dict[str, str]]:
    """Return (primary_annotation, merged_values).

    In absolute mode, delete additional annotations in our namespace after merging.
    In non-absolute mode, keep them but still merge their values for the update logic.
    """
    annotations = get_map_annotations(obj)
    if not annotations:
        return None, {}

    primary = annotations[0]
    merged: Dict[str, str] = {}
    for ann in annotations:
        merged.update(map_annotation_to_dict(ann))

    if absolute and len(annotations) > 1:
        extras = annotations[1:]
        try:
            conn.deleteObjects("Annotation", [ann.getId() for ann in extras], wait=True)
            logging.info("Removed %d duplicate map annotations (absolute)", len(extras))
        except Exception as exc:
            logging.warning("Failed to delete duplicate map annotations: %s", exc)

    return primary, merged


def merge_key_values(existing: Dict[str, str], additions: Dict[str, str]) -> Tuple[Dict[str, str], bool]:
    """Merge additions into existing and report whether anything changed."""
    merged = existing.copy()
    changed = False
    for key, value in additions.items():
        new_value = str(value)
        if merged.get(key) != new_value:
            merged[key] = new_value
            changed = True
    return merged, changed


def dict_to_key_value_list(data: Dict[str, str]) -> List[List[str]]:
    return [[str(k), str(v)] for k, v in data.items()]


def describe_object(obj) -> str:
    try:
        obj_type = obj.getObjectType()
    except Exception:
        obj_type = "object"
    try:
        name = obj.getName()
    except Exception:
        name = "<unnamed>"
    try:
        obj_id = obj.getId()
    except Exception:
        obj_id = "?"
    return f"{obj_type} '{name}' (ID {obj_id})"


def add_map_annotation(conn: BlitzGateway, obj, key_values: Dict[str, str], absolute: bool = False) -> None:
    """Add or update map annotation on an OMERO object.

    absolute=False (default): add or modify keys; do not remove existing keys or annotations.
    absolute=True: treat CSV as source of truth; remove keys not present and delete extra annotations in our namespace.
    """
    filtered_kv = {
        k: v
        for k, v in key_values.items()
        if k not in ["subject_id", "sample_id"] and str(v).strip() != ""
    }

    if not filtered_kv:
        logging.info("No additional metadata to add")
        return

    existing, existing_values = consolidate_map_annotations(conn, obj, absolute=absolute)
    if absolute:
        merged = filtered_kv
        changed = existing_values != merged
    else:
        merged, changed = merge_key_values(existing_values, filtered_kv)

    if not changed:
        logging.info("Metadata already up to date (no changes) for %s", describe_object(obj))
        return

    key_value_data = dict_to_key_value_list(merged)
    if existing:
        existing.setValue(key_value_data)
        existing.save()
        logging.info("Updated map annotation (%d entries)%s", len(merged), " (absolute)" if absolute else "")
    else:
        map_ann = MapAnnotationWrapper(conn)
        map_ann.setNs(MAP_ANNOTATION_NS)
        map_ann.setValue(key_value_data)
        map_ann.save()
        obj.linkAnnotation(map_ann)
        logging.info("Created map annotation (%d entries)%s", len(merged), " (absolute)" if absolute else "")


def find_dataset_by_name(conn: BlitzGateway, subject_id: str) -> Optional[DatasetWrapper]:
    """Find a dataset by name (subject_id)."""
    datasets = list(conn.getObjects("Dataset", attributes={"name": subject_id}))
    if len(datasets) == 1:
        return datasets[0]
    if len(datasets) > 1:
        logging.warning("Multiple datasets matched name '%s' - skipping", subject_id)
        return None
    return None


def find_image_candidates(conn: BlitzGateway, sample_id: str) -> List[ImageWrapper]:
    """Return list of images matching sample_id.

    Strategy:
    - Try exact name match first
    - If none found, fall back to substring match on image name
    """
    # exact matches
    exact = list(conn.getObjects("Image", attributes={"name": sample_id}))
    if exact:
        return exact

    # fallback: substring match (scan images)
    candidates: List[ImageWrapper] = []
    for img in conn.getObjects("Image"):
        try:
            if sample_id in img.getName():
                candidates.append(img)
        except Exception:
            # skip objects we can't inspect
            continue

    return candidates


def _filtered_kv_from_row(row: Dict[str, str]) -> Dict[str, str]:
    return {
        k: v
        for k, v in row.items()
        if k not in ["subject_id", "sample_id"] and str(v).strip() != ""
    }


def resolve_targets(conn: BlitzGateway, rows: List[Dict[str, str]]):
    """Resolve each row to a single OMERO object when possible.

    Returns:
      - resolved_map: dict[row_index] = (obj_type, obj_id, obj_name)
      - duplicates: dict[(obj_type, obj_id)] = list[row_indices]
    Row indices are 2-based to match CSV line numbers (header is 1).
    """
    resolved_map = {}
    obj_to_rows = {}

    for idx, row in enumerate(rows, start=2):
        subject_id = row.get("subject_id", "").strip()
        sample_id = row.get("sample_id", "").strip()

        target = None
        # prioritize sample_id
        if sample_id:
            candidates = find_image_candidates(conn, sample_id)
            if len(candidates) == 1:
                img = candidates[0]
                target = ("image", img.getId(), img.getName())
            elif len(candidates) == 0:
                logging.warning("Row %d: image not found for sample_id '%s'", idx, sample_id)
            else:
                logging.warning("Row %d: sample_id '%s' matched multiple images (%d) - skipping", idx, sample_id, len(candidates))
        elif subject_id:
            ds = find_dataset_by_name(conn, subject_id)
            if ds:
                target = ("dataset", ds.getId(), ds.getName())
            else:
                logging.warning("Row %d: dataset not found for subject_id '%s'", idx, subject_id)

        if target:
            resolved_map[idx] = target
            key = (target[0], target[1])
            obj_to_rows.setdefault(key, []).append(idx)

    duplicates = {k: v for k, v in obj_to_rows.items() if len(v) > 1}
    return resolved_map, duplicates


def analyze_row_conflicts(rows: List[Dict[str, str]], row_idxs: List[int]) -> List[str]:
    """Return list of keys that have conflicting values across the given rows.

    row_idxs are CSV line numbers (2-based). Index into rows list by subtracting 2.
    """
    key_values = {}
    conflicts = set()
    for row_idx in row_idxs:
        row = rows[row_idx - 2]
        filtered = _filtered_kv_from_row(row)
        for k, v in filtered.items():
            v_str = str(v)
            if k in key_values and key_values[k] != v_str:
                conflicts.add(k)
            key_values.setdefault(k, v_str)
    return sorted(conflicts)


def process_csv_row(conn: BlitzGateway, row: Dict[str, str], row_num: int, absolute: bool) -> None:
    """Process a single CSV row and assign metadata to OMERO."""
    subject_id = row.get('subject_id', '').strip()
    sample_id = row.get('sample_id', '').strip()
    
    # Prioritize sample_id if both present
    if sample_id:
        logging.info("Row %d: Processing sample_id '%s'", row_num, sample_id)
        images = find_image_candidates(conn, sample_id)
        if len(images) == 1:
            image = images[0]
            logging.info("Found image: %s (ID: %s)", image.getName(), image.getId())
            add_map_annotation(conn, image, row, absolute=absolute)
        elif len(images) == 0:
            logging.warning("Image not found for sample_id '%s'", sample_id)
        else:
            logging.warning("Multiple images matched sample_id '%s' (%d matches) - skipping", sample_id, len(images))
    elif subject_id:
        logging.info("Row %d: Processing subject_id '%s'", row_num, subject_id)
        dataset = find_dataset_by_name(conn, subject_id)
        if dataset:
            logging.info("Found dataset: %s (ID: %s)", dataset.getName(), dataset.getId())
            add_map_annotation(conn, dataset, row, absolute=absolute)
        else:
            logging.warning("Dataset not found for subject_id '%s'", subject_id)
    else:
        logging.warning("Row %d: No subject_id or sample_id found, skipping", row_num)


def process_csv(csv_file: str, conn_params, absolute: bool) -> None:
    """Read CSV file and process each row."""
    conn = None
    try:
        host, port, user, password, secure = conn_params
        conn = connect_to_omero(host, port, user, password, secure)
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            if not reader.fieldnames:
                raise ValueError("CSV file appears to be empty or invalid")

            rows = list(reader)
            if not rows:
                logging.info("CSV contains header but no data rows")
                return

            logging.info("Processing CSV file: %s", csv_file)
            logging.info("Columns: %s", ", ".join(reader.fieldnames))

            # Pre-scan: resolve targets and detect duplicate object targets
            resolved_map, duplicates = resolve_targets(conn, rows)
            if duplicates:
                # Report duplicates with guidance
                for obj_key, row_idxs in duplicates.items():
                    obj_type, obj_id = obj_key
                    rows_str = ", ".join(str(r) for r in row_idxs)
                    if absolute:
                        logging.warning(
                            "%s %s is defined in multiple CSV rows (%s). "
                            "Only the last row will be applied in --absolute mode.",
                            obj_type.capitalize(), obj_id, rows_str,
                        )
                    else:
                        # In additive mode we warn and check for conflicting key-values
                        conflicts = analyze_row_conflicts(rows, row_idxs)
                        if conflicts:
                            logging.warning(
                                "%s %s is defined in multiple CSV rows (%s). Conflicting keys: %s. "
                                "In additive mode only mismatching keys may cause issues; last write wins for conflicts.",
                                obj_type.capitalize(), obj_id, rows_str, ", ".join(conflicts),
                            )
                        else:
                            logging.warning(
                                "%s %s is defined in multiple CSV rows (%s). Rows are additive and non-conflicting.",
                                obj_type.capitalize(), obj_id, rows_str,
                            )

            for idx, row in enumerate(rows, start=2):  # Start at 2 (header is row 1)
                process_csv_row(conn, row, idx, absolute)

        logging.info("Processing complete!")

    except Exception as e:
        logging.error("Error: %s", e)
        sys.exit(1)
    finally:
        if conn:
            conn.close()
            logging.info("Disconnected from OMERO")


def main():
    args = parse_args()

    # configure logging
    level = logging.INFO if getattr(args, "verbose", False) else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    if not os.path.exists(args.csv_file):
        logging.error("File not found: %s", args.csv_file)
        sys.exit(1)

    conn_params = build_connection_config(args)
    process_csv(args.csv_file, conn_params, args.absolute)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for CSV processing and OMERO connection."""
    parser = argparse.ArgumentParser(
        description=(
            "Assign CSV metadata to OMERO datasets (subject_id) or images (sample_id). "
            "If both are present, sample_id takes priority."
        )
    )
    parser.add_argument("csv_file", help="Path to CSV file containing metadata")
    parser.add_argument("--host", help="OMERO host (defaults to OMERO_HOST or OMERO_SERVER env vars)")
    parser.add_argument("--port", type=int, help="OMERO port (defaults to OMERO_PORT or 4064)")
    parser.add_argument("--user", help="OMERO username (defaults to OMERO_USER env var)")
    parser.add_argument("--password", help="OMERO password (defaults to OMERO_PASSWORD env var)")
    parser.add_argument(
        "--secure",
        action="store_true",
        help="Force secure connection (default if neither secure nor insecure is set)",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Force insecure connection",
    )
    parser.add_argument(
        "--absolute",
        action="store_true",
        help="Treat CSV as source of truth: remove keys not present and delete duplicate map annotations",
    )

    # add verbose flag here for controlling logging
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable informational logging")

    args = parser.parse_args()
    if args.secure and args.insecure:
        parser.error("--secure and --insecure are mutually exclusive")

    return args


def build_connection_config(args: argparse.Namespace):
    """Build OMERO connection parameters from CLI args and environment."""
    host_env = os.environ.get("OMERO_HOST") or os.environ.get("OMERO_SERVER")
    host = args.host or host_env
    port = args.port or int(os.environ.get("OMERO_PORT", "4064"))
    user = args.user or os.environ.get("OMERO_USER")
    password = args.password or os.environ.get("OMERO_PASSWORD")
    secure = True if args.secure else False if args.insecure else True

    missing = [name for name, val in [("host", host), ("user", user), ("password", password)] if not val]
    if missing:
        logging.error("Missing required connection parameters: %s", ", ".join(missing))
        sys.exit(1)

    # Type narrowing for static checkers
    return str(host), int(port), str(user), str(password), bool(secure)


if __name__ == "__main__":
    main()
