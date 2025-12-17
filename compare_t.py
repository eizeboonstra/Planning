import argparse
import csv
from typing import Dict, Tuple


def _parse_id_token(tok):
    try:
        xi = float(tok)
        return int(xi) if xi.is_integer() else tok
    except Exception:
        try:
            return int(tok)
        except Exception:
            return tok


def _parse_var_name(name: str):
    # Expect formats like "t_12_34" or "t_12.0_34.0"
    parts = name.strip().split("_")
    if len(parts) >= 3 and parts[0].lower() == "t":
        p = _parse_id_token(parts[1])
        r = _parse_id_token(parts[2])
        return p, r
    return None, None


def load_t_csv(path: str) -> Dict[Tuple[int, int], float]:
    mapping: Dict[Tuple[int, int], float] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = None
            r = None
            val = None

            # Preferred columns
            if {"p_id", "r_id", "t_value"}.issubset(row.keys()):
                p = _parse_id_token(row["p_id"]) if row["p_id"] != "" else None
                r = _parse_id_token(row["r_id"]) if row["r_id"] != "" else None
                try:
                    val = float(row["t_value"]) if row["t_value"] != "" else 0.0
                except Exception:
                    val = 0.0
            else:
                # Try generic VarName/value style
                name_key = None
                for k in ("VarName", "varName", "name", "var"):
                    if k in row:
                        name_key = k
                        break
                value_key = None
                for k in ("t_value", "Value", "value", "val", "X"):
                    if k in row:
                        value_key = k
                        break
                if name_key and value_key:
                    p, r = _parse_var_name(row[name_key])
                    try:
                        val = float(row[value_key]) if row[value_key] != "" else 0.0
                    except Exception:
                        val = 0.0

            if p is None or r is None or val is None:
                continue
            try:
                p_i = int(p)
                r_i = int(r)
            except Exception:
                # Skip non-integer IDs
                continue
            mapping[(p_i, r_i)] = mapping.get((p_i, r_i), 0.0) + float(val)
    return mapping


def main():
    ap = argparse.ArgumentParser(description="Compare t variables between problem2 and expensive model.")
    ap.add_argument("--problem2", "-p", default="t_vars_problem2.csv", help="CSV from problem2 with columns p_id,r_id,t_value or VarName/value")
    ap.add_argument("--expensive", "-e", default="t_vars_expensive_derived.csv", help="CSV derived from expensive_model")
    ap.add_argument("--tol", type=float, default=1e-6, help="Tolerance for mismatch")
    ap.add_argument("--ignore-dummy", type=int, default=382, help="Ignore rows with r_id equal to this value (set -1 to disable)")
    ap.add_argument("--out", default="", help="Optional output CSV of mismatches")
    args = ap.parse_args()

    p2 = load_t_csv(args.problem2)
    exp = load_t_csv(args.expensive)

    keys = set(p2.keys()) | set(exp.keys())
    mismatches = []
    ignored = 0
    for k in sorted(keys):
        p_id, r_id = k
        if args.ignore_dummy >= 0 and r_id == args.ignore_dummy:
            ignored += 1
            continue
        v1 = p2.get(k, 0.0)
        v2 = exp.get(k, 0.0)
        diff = v1 - v2
        if abs(diff) > args.tol:
            mismatches.append({"p_id": p_id, "r_id": r_id, "t_problem2": v1, "t_expensive": v2, "diff": diff})

    print(f"Compared {len(keys)} pairs (ignored {ignored} with r_id == {args.ignore_dummy}).")
    print(f"Mismatches > {args.tol}: {len(mismatches)}")
    if mismatches:
        # Show a few examples
        for row in mismatches[:20]:
            print(f"(p={row['p_id']}, r={row['r_id']}): p2={row['t_problem2']:.6f}, exp={row['t_expensive']:.6f}, diff={row['diff']:.6f}")

    if args.out:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["p_id", "r_id", "t_problem2", "t_expensive", "diff"]) 
            writer.writeheader()
            for row in mismatches:
                writer.writerow(row)
        print(f"Wrote mismatches to {args.out}")


if __name__ == "__main__":
    main()
