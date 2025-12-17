import math
import gurobipy as gp
from gurobipy import GRB
from sets import get_flights, get_itineraries, get_recapture


# =========================
# Utility
# =========================
RC_TOL = -1e-6

def is_nan(x):
    return isinstance(x, float) and math.isnan(x)

def clean_flights_pair(f1, f2):
    out = []
    for v in (f1, f2):
        if v is None or v == 0 or is_nan(v):
            continue
        # pandas spesso legge come float (102.0)
        out.append(int(v))
    return out

def fmt_num(x):
    # stampa "12" invece di "12.0", ma "63.2" con 1 decimale
    if abs(x - round(x)) < 1e-6:
        return str(int(round(x)))
    return f"{x:.1f}"

def dummy_label(dummy_id):
    # Nelle slide il dummy è 0; nel tuo sets.py è 382
    return 0

def print_rmp_iteration(iter_no, model, cap_constr, dem_constr, dummy_id, prev_obj=None):
    obj = model.objVal
    delta_str = ""
    if prev_obj is not None:
        delta = obj - prev_obj
        # slide mostra tipo (-1574)
        delta_str = f" ({int(round(delta))})"
    print(f"\nRMP - Iteration {iter_no}:")
    print(f"  OF Value = {int(round(obj))}{delta_str}")

    # Decision variables non null
    print("\nDecision variables (non null):")
    nz = []
    for v in model.getVars():
        if abs(v.X) > 1e-6 and v.VarName.startswith("t_"):
            _, p, r = v.VarName.split("_")
            p = int(p); r = int(r)
            r_print = dummy_label(dummy_id) if r == dummy_id else r
            nz.append((p, r_print, v.X))
    nz.sort(key=lambda t: (t[0], t[1]))
    for p, r, val in nz:
        print(f"  t_{p}^{r} = {fmt_num(val)}")

    # Dual variables non null
    print("\nDual variables (non null):")
    pis = []
    for fid, c in cap_constr.items():
        if abs(c.Pi) > 1e-6:
            pis.append((int(fid), c.Pi))
    pis.sort(key=lambda x: x[0])
    for fid, pi in pis:
        # se ti uscisse il segno opposto per convenzione, commenta la riga sotto e usa "-pi"
        print(f"  pi_{fid} = {fmt_num(pi)}")

    sigmas = []
    for pid, c in dem_constr.items():
        if abs(c.Pi) > 1e-6:
            # non stampare il dummy
            if pid == dummy_id:
                continue
            sigmas.append((int(pid), c.Pi))
    sigmas.sort(key=lambda x: x[0])
    for pid, s in sigmas:
        print(f"  sigma_{pid} = {fmt_num(s)}")


# =========================
# Main: Column Generation (LP) con output stile slide
# =========================
def run_column_generation_like_slides():
    flights = get_flights()
    itins = get_itineraries()
    recaps = get_recapture()

    # Dummy id dal tuo sets.py
    dummy_id = 382 if any(it.itinerary_id == 382 for it in itins) else 0

    # Mappe
    fare = {it.itinerary_id: float(it.price) for it in itins}
    demand = {it.itinerary_id: float(it.demand) for it in itins}
    itin_flights = {it.itinerary_id: clean_flights_pair(it.flight1, it.flight2) for it in itins}

    # Q_i = domanda originale su ogni volo i
    Q = {f.flight_id: 0.0 for f in flights}
    for it in itins:
        if it.itinerary_id == dummy_id:
            continue
        for fid in itin_flights.get(it.itinerary_id, []):
            if fid in Q:
                Q[fid] += float(it.demand)

    # Recapture map (solo quelle presenti in sheet + dummy arcs)
    b = {(r.from_itinerary, r.to_itinerary): float(r.recapture_rate) for r in recaps}

    # Insiemi archi:
    # - spill arcs: (p -> dummy)
    # - candidate arcs: (p -> r) con r != dummy
    spill_arcs = []
    cand_arcs = []
    for r in recaps:
        p = r.from_itinerary
        q = r.to_itinerary
        if p == dummy_id:
            continue
        if q == dummy_id:
            spill_arcs.append((p, q, float(r.recapture_rate)))
        else:
            cand_arcs.append((p, q, float(r.recapture_rate)))

    # Rimuovi duplicati (sets.py aggiunge anche dummy->dummy ecc.)
    spill_arcs = list({(p, q): rate for (p, q, rate) in spill_arcs}.items())
    spill_arcs = [(p, q, rate) for ((p, q), rate) in spill_arcs]

    cand_arcs = list({(p, q): rate for (p, q, rate) in cand_arcs}.items())
    cand_arcs = [(p, q, rate) for ((p, q), rate) in cand_arcs]

    # =========================
    # Build RMP with fixed constraints
    # =========================
    m = gp.Model("RMP_Column_Generation_LP")
    # per replicare bene dual/soluzione (utile per confrontare slide)
    m.Params.OutputFlag = 0
    m.Params.Method = 1      # dual simplex
    m.Params.Presolve = 0

    # Capacity constraints: expr >= Q_i - CAP_i
    cap_constr = {}
    for f in flights:
        rhs = Q[f.flight_id] - float(f.capacity)
        cap_constr[f.flight_id] = m.addConstr(gp.LinExpr() >= rhs, name=f"C1_Capacity_{f.flight_id}")

    # Demand constraints: sum_r t_{p,r} <= demand_p
    dem_constr = {}
    for it in itins:
        pid = it.itinerary_id
        if pid == dummy_id:
            continue
        dem_constr[pid] = m.addConstr(gp.LinExpr() <= demand[pid], name=f"C2_Demand_{pid}")

    m.update()

    # helper: add column for variable t_{p,r}
    active = set()

    def add_var(p, r, b_pr):
        if (p, r) in active:
            return
        if p == dummy_id:
            return
        # Objective coeff: c_{p,r} = fare_p - b_{p,r} fare_r
        fare_p = fare.get(p, 0.0)
        fare_r = fare.get(r, 0.0)
        c = fare_p - b_pr * fare_r

        flights_p = set(itin_flights.get(p, []))
        flights_r = set(itin_flights.get(r, []))
        all_f = flights_p | flights_r

        col = gp.Column()

        # Coefficiente su C1 (equivalente alla forma "variabile invertita"):
        # a_i(p,r) = δ_i^p - b_{p,r} δ_i^r
        for fid in all_f:
            if fid not in cap_constr:
                continue
            delta_p = 1.0 if fid in flights_p else 0.0
            delta_r = 1.0 if fid in flights_r else 0.0
            a = delta_p - b_pr * delta_r
            if abs(a) > 1e-12:
                col.addTerms(a, cap_constr[fid])

        # Coefficiente su C2 (demand di p): +1
        if p in dem_constr:
            col.addTerms(1.0, dem_constr[p])

        m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, obj=c, name=f"t_{p}_{r}", column=col)
        m.update()
        active.add((p, r))

    # =========================
    # Initial columns: ONLY spill (p -> dummy)
    # =========================
    for (p, r, rate) in spill_arcs:
        if r != dummy_id:
            continue
        add_var(p, r, 1.0)  # spill: b=1, fare_dummy=0

    # =========================
    # Column Generation loop (aggiungo TUTTE le colonne con RC<0 in un colpo)
    # per replicare l’idea delle slide: Iteration 1 (spill only), Iteration 2 (dopo aggiunta colonne)
    # =========================
    prev_obj = None
    iter_no = 1

    while True:
        m.optimize()
        if m.status != GRB.OPTIMAL:
            print("RMP non ottimo. Status:", m.status)
            return

        print_rmp_iteration(iter_no, m, cap_constr, dem_constr, dummy_id, prev_obj=prev_obj)
        prev_obj = m.objVal

        # Duals
        pi = {fid: cap_constr[fid].Pi for fid in cap_constr}
        sigma = {pid: dem_constr[pid].Pi for pid in dem_constr}

        # Pricing
        improving = []
        best_rc = float("inf")

        for (p, r, rate) in cand_arcs:
            if p == dummy_id or r == dummy_id:
                continue
            if (p, r) in active:
                continue

            b_pr = b.get((p, r), 0.0)
            fare_p = fare.get(p, 0.0)
            fare_r = fare.get(r, 0.0)
            c = fare_p - b_pr * fare_r

            flights_p = set(itin_flights.get(p, []))
            flights_r = set(itin_flights.get(r, []))
            all_f = flights_p | flights_r

            dual_sum = 0.0
            for fid in all_f:
                if fid not in pi:
                    continue
                delta_p = 1.0 if fid in flights_p else 0.0
                delta_r = 1.0 if fid in flights_r else 0.0
                a = delta_p - b_pr * delta_r
                dual_sum += a * pi[fid]

            rc = c - dual_sum - sigma.get(p, 0.0)
            best_rc = min(best_rc, rc)
            if rc < RC_TOL:
                improving.append((p, r, b_pr, rc))

        if not improving:
            print("\nNo more columns to be added  ->  Optimal solution")
            break

        # Aggiungo TUTTE le colonne con RC<0 (come spesso fatto negli esempi didattici)
        for (p, r, b_pr, rc) in sorted(improving, key=lambda x: x[3]):
            add_var(p, r, b_pr)

        iter_no += 1


if __name__ == "__main__":
    run_column_generation_like_slides()

