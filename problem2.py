import gurobipy as gp
from gurobipy import GRB

import pandas as pd
import math
EPS = 1e-6
DUMMY_ITIN_ID = 382

data_file = "C:\\TUDELFT\\Sustainable Air Transport\\Q2\\airline_planning\\Problem 2 - Data\\Problem 2 - Data\\Group_11.xlsx"
xls = pd.ExcelFile(data_file)  # ExcelFile is not subscriptable; pass it to pd.read_excel

# Passenger Mix Flow problem
model = gp.Model("Passenger_Mix_Flow")

# Sets

class Flights:
    def __init__(self, flight_id, origin, destination,departure, ready, capacity):
        self.flight_id = flight_id
        self.origin = origin
        self.destination = destination
        self.departure = departure
        self.ready = ready
        self.capacity = capacity

class Itineraries:
    def __init__(self, itinerary_id, origin, flight1, flight2, destination, demand, price):
        self.itinerary_id = itinerary_id
        self.origin = origin
        self.flight1 = flight1
        self.flight2 = flight2
        self.destination = destination
        self.price = price
        self.demand = demand

class Recapture:
    def __init__(self, from_itinerary, to_itinerary, recapture_rate):
        self.from_itinerary = from_itinerary
        self.to_itinerary = to_itinerary
        self.recapture_rate = recapture_rate

def get_flights():
    df = pd.read_excel(xls, sheet_name="Flights")  # header row assumed
    flights = []
    for r in df.itertuples(index=False):  # position-based to avoid column name mismatches
        flight = Flights(
            flight_id=r[0],
            origin=r[1],
            destination=r[2],
            departure=r[3],
            ready=r[4],
            capacity=r[5]
        )
        flights.append(flight)
    return flights

def get_itineraries():
    df = pd.read_excel(xls, sheet_name="Itineraries")
    itineraries = []
    for r in df.itertuples(index=False):
        itinerary = Itineraries(
            itinerary_id=r[0],
            origin=r[1],
            destination=r[2],
            flight1=r[3],
            flight2=r[4],
            price=r[5],
            demand=r[6]
        )
        itineraries.append(itinerary)

    # Add a dummy itinerary with fare 0
    dummy_itinerary = Itineraries(
        itinerary_id=382,
        origin=None,
        destination=None,
        flight1=None,
        flight2=None,
        price=0,
        demand=0
    ) 
    itineraries.append(dummy_itinerary)

    return itineraries

def get_recapture():
    df = pd.read_excel(xls, sheet_name="Recapture")
    recaptures = []
    for r in df.itertuples(index=False):
        recapture = Recapture(
            from_itinerary=r[0],
            to_itinerary=r[1],
            recapture_rate=r[2]
        )
        recaptures.append(recapture)
    
    for itin in get_itineraries():
        # Add recapture to dummy itinerary
        recapture = Recapture(
            from_itinerary=itin.itinerary_id,
            to_itinerary=382,
            recapture_rate=1.0
        )
        recaptures.append(recapture)

    return recaptures

# simple smoke test
flights = get_flights()
itins = get_itineraries()
recaps = get_recapture()
#=============================================================================================================================

# create variables

#number of passangers that would like p but are reallocated to r 

for r in recaps:
    p_id = r.from_itinerary
    r_id = r.to_itinerary
    var_name = f"t_{p_id}_{r_id}"
    model.addVar(vtype=GRB.INTEGER, name=var_name, lb=0)

model.update()
#=============================================================================================================================
# constraints
#=============================================================================================================================
# 1. Define the necessary helper function (as defined in the previous turn)
def get_flight_total_demand(flights, itins):
    """Calculates the total original demand (Q_i) for each flight leg i."""
    flight_demand = {f.flight_id: 0 for f in flights}
    for flight in flights:
        for itin in itins:
            if flight.flight_id == itin.flight1 or flight.flight_id == itin.flight2:
                flight_demand[flight.flight_id] += itin.demand
    return flight_demand


def build_model_constraints(model, flights, itins, recaps):
    # Data lookups for recapture rates
    recapture_map = {(r.from_itinerary, r.to_itinerary): r.recapture_rate for r in recaps}
    itin_flights = {itin.itinerary_id: [itin.flight1, itin.flight2] for itin in itins}
    demand_map = {itin.itinerary_id: itin.demand for itin in itins}

    # Pre-calculate Q_i for C1
    flight_demand_Q = get_flight_total_demand(flights, itins)

    # --- C1: Capacity Constraints ---
    for flight in flights:
        flight_id = flight.flight_id
        constr_expr = gp.LinExpr()

        # Term 1:
        for recap in recaps:
            p_id = recap.from_itinerary
            r_id = recap.to_itinerary

            # Check if flight_id is in itinerary p
            if flight_id in itin_flights.get(p_id, []):
                var_name = f"t_{p_id}_{r_id}"
                t_var = model.getVarByName(var_name)
                if t_var is not None:
                    constr_expr += t_var

        # Term 2:
        for recap in recaps:
            p_id = recap.from_itinerary
            r_id = recap.to_itinerary
            b_pr = recap.recapture_rate

            # Check if flight_id is in itinerary r
            if flight_id in itin_flights.get(r_id, []):
                var_name = f"t_{r_id}_{p_id}"
                t_var = model.getVarByName(var_name)
                if t_var is not None:
                    constr_expr -= b_pr * t_var

        # RHS: use original difference Q_i - CAP_i (do NOT clamp to 0)
        rhs = flight_demand_Q.get(flight_id, 0) - flight.capacity

        if rhs > 0 and constr_expr.size() == 0:
            continue  # recapture not possible for this flight, skip adding constraint

        # Always add the constraint (if constr_expr is zero this becomes 0 >= rhs)
        model.addConstr(constr_expr >= rhs, name=f"C1_Capacity_{flight_id}")

    # --- C2: Demand Constraints ---
    for p_itin in itins:
        p_id = p_itin.itinerary_id
        constr_expr = gp.LinExpr()
        for recap in recaps:
            if recap.from_itinerary == p_id:
                var = model.getVarByName(f"t_{p_id}_{recap.to_itinerary}")
                if var is not None:
                    constr_expr += var
        rhs = demand_map.get(p_id, 0)
        model.addConstr(constr_expr <= rhs, name=f"C2_Demand_{p_id}")

    # C3 (Non-negativity) handled by lb=0
    model.update()
    return model


def set_objective_function(model, itins, recaps):
    # Data lookups for prices (fares)
    fare_map = {itin.itinerary_id: itin.price for itin in itins}
    recapture_map = {(r.from_itinerary, r.to_itinerary): r.recapture_rate for r in recaps}

    obj_expr = gp.LinExpr()

    for r in recaps:
        p_id = r.from_itinerary
        r_id = r.to_itinerary
        
        # Get variable t_{p_id}_{r_id}
        var_name = f"t_{p_id}_{r_id}"
        t_var = model.getVarByName(var_name)

        if t_var:
            fare_p = fare_map.get(p_id, 0)
            fare_r = fare_map.get(r_id, 0)
            b_pr = recapture_map.get((p_id, r_id), 0)
            
            # Cost coefficient: (fare_p - b_p^r * fare_r)
            cost_coeff = fare_p - b_pr * fare_r
            
            obj_expr += cost_coeff * t_var
    
    model.setObjective(obj_expr, GRB.MINIMIZE)
    model.update()
    return model

def calculate_reduced_cost(model, flights, itins, p_id, r_id, b_pr):
    """
    Calculates the reduced cost for a potential new column (t_{p,r}).
    Requires the Master Problem (model) to be solved first.
    """
    # 1. Get Dual Variables (Duals are only available after model.optimize())
    pi_map = {c.ConstrName.split('_')[-1]: c.Pi for c in model.getConstrs() if c.ConstrName.startswith('C1_Capacity')}
    sigma_map = {c.ConstrName.split('_')[-1]: c.Pi for c in model.getConstrs() if c.ConstrName.startswith('C2_Demand')}
    
    # 2. Get data for p and r
    p_itin = next((i for i in itins if i.itinerary_id == p_id), None)
    r_itin = next((i for i in itins if i.itinerary_id == r_id), None)

    if not p_itin or not r_itin:
        print(f"Error: Itinerary {p_id} or {r_id} not found.")
        return float('inf')

    fare_p = p_itin.price
    fare_r = r_itin.price
    
    # 3. Calculate Summation Term (Reduced Cost from C1)
    sum_term = 0
    p_flights = [p_itin.flight1, p_itin.flight2]
    r_flights = [r_itin.flight1, r_itin.flight2]

    # Iterate over all flights (L) to calculate the sum
    for flight in flights:
        flight_id = flight.flight_id
        pi_i = pi_map.get(flight_id, 0.0)

        # delta_i^p: 1 if flight i is used by p
        delta_ip = 1 if flight_id in p_flights else 0
        # delta_i^r: 1 if flight i is used by r
        delta_ir = 1 if flight_id in r_flights else 0

        sum_term += (delta_ip - delta_ir * b_pr) * pi_i

    # 4. Get Sigma Term (Reduced Cost from C2)
    sigma_p = sigma_map.get(p_id, 0.0)

    # 5. Calculate Final Reduced Cost
    # Reduced Cost = Sum_Term + Sigma_p - (Cost Coefficient)
    cost_coeff = fare_p - b_pr * fare_r
    reduced_cost = sum_term + sigma_p - cost_coeff
    
    return reduced_cost


model.setParam("LogFile", "passenger_mix_flow.log")  # SAVE LOG TO FILE

# Add constraints
build_model_constraints(model, flights, itins, recaps)

# Set objective function
set_objective_function(model, itins, recaps)

# Write model as LP file
model.write("PassengerMixFlow.lp")

# Optimize model
model.optimize()

# If solve not optimal, compute IIS and write it out for inspection
if model.status != GRB.OPTIMAL:
    print("\nModel not optimal. Status code:", model.status)
    try:
        model.computeIIS()
        model.write("PassengerMixFlow_infeasible.ilp")
        print("IIS written to PassengerMixFlow_infeasible.ilp. Constraints in IIS:")
        for c in model.getConstrs():
            if c.IISConstr:
                print(c.ConstrName)
    except gp.GurobiError as e:
        print("IIS computation failed:", e)

# =============================================================
# CHECK RESULTS
# =============================================================

if model.status == GRB.OPTIMAL:
    print("\n=== OPTIMAL SOLUTION FOUND ===")
    print(f"Optimal objective value = {model.objVal}")

    print("\n--- Decision Variables ---")
    for v in model.getVars():
        if abs(v.X) > 1e-6:  # print only nonzero variables
            print(f"{v.VarName} = {v.X}")

else:
    print("\nModel did not solve to optimality.")
    print(f"Gurobi Status Code: {model.status}")


#=============================================================================================================================
#  Implementation of Column Generation
#=============================================================================================================================

# def create_master_model(flights, itins, active_recaps, var_type=GRB.CONTINUOUS):
#     """Build a fresh master model containing only active_recaps (list of Recapture objects)."""
#     m = gp.Model("Master_RMP")
#     m.setParam("OutputFlag", 0)  # turn off Gurobi printing for iterations
#     # create variables for each active recap
#     for r in active_recaps:
#         var_name = f"t_{r.from_itinerary}_{r.to_itinerary}"
#         m.addVar(vtype=var_type, name=var_name, lb=0)
#     m.update()
#     # add constraints & objective using existing helper functions (they read vars by name)
#     build_model_constraints(m, flights, itins, active_recaps)
#     set_objective_function(m, itins, active_recaps)
#     return m

# def initial_active_recaps(all_recaps):
#     """Start with columns that recapture to the dummy itinerary only."""
#     active = [r for r in all_recaps if r.to_itinerary == DUMMY_ITIN_ID]
#     # If none exist (defensive), add a dummy for every p->dummy
#     if not active:
#         p_ids = {r.from_itinerary for r in all_recaps}
#         active = [Recapture(from_itinerary=p, to_itinerary=DUMMY_ITIN_ID, recapture_rate=1.0) for p in p_ids]
#     return active

# def recap_key(r):
#     return (r.from_itinerary, r.to_itinerary)

# def active_keys(active_recaps):
#     return {recap_key(r) for r in active_recaps}

# def column_generation(flights, itins, all_recaps, max_iters=200):
#     """Column generation loop. Returns final active_recaps and solution model (LP)."""
#     active = initial_active_recaps(all_recaps)
#     all_keys = {(r.from_itinerary, r.to_itinerary): r for r in all_recaps}
#     iter_count = 0

#     while True:
#         iter_count += 1
#         if iter_count > max_iters:
#             print("Max CG iterations reached.")
#             break

#         # Build and solve restricted master (LP)
#         rmp = create_master_model(flights, itins, active, var_type=GRB.INTEGER)
#         rmp.optimize()
#         if rmp.status != GRB.OPTIMAL:
#             print("RMP not optimal or infeasible during CG. Status:", rmp.status)
#             break

#         # Compute reduced costs for all candidate recaptures not yet in active set
#         candidates = []
#         active_set = active_keys(active)
#         for key, rec in all_keys.items():
#             if key in active_set:
#                 continue
#             p_id, r_id = key
#             b_pr = rec.recapture_rate
#             rc = calculate_reduced_cost(rmp, flights, itins, p_id, r_id, b_pr)
#             candidates.append((rc, rec))

#         if not candidates:
#             break

#         # find best (most negative) reduced cost
#         best_rc, best_rec = min(candidates, key=lambda x: x[0])

#         if best_rc < -EPS:
#             active.append(best_rec)
#             print(f"Iter {iter_count}: Adding column t_{best_rec.from_itinerary}_{best_rec.to_itinerary} (rc={best_rc:.6f})")
#         else:
#             print(f"Iter {iter_count}: No improving columns (best reduced cost = {best_rc:.6f}).")
#             break

#     # return last RMP and active_recaps (RMP currently solved)
#     return active, rmp

# # === Main CG driver ===
# # Build full recapture list once (all potential columns)
# full_recaps = get_recapture()

# # Run column generation (LP relaxation)
# active_recaps, final_lp = column_generation(flights, itins, full_recaps)

# # final_lp contains the LP solution with active columns. If you need integer solution:
# # 1) Rebuild master with integer variables for active_recaps and solve as MIP.
# final_mip = create_master_model(flights, itins, active_recaps, var_type=GRB.INTEGER)
# final_mip.setParam("LogFile", "PassengerMixFlow_final_mip.log")
# final_mip.optimize()

# if final_mip.status == GRB.OPTIMAL:
#     print("\n=== FINAL INTEGER SOLUTION ===")
#     print(f"Objective = {final_mip.objVal}")
#     for v in final_mip.getVars():
#         if abs(v.X) > 1e-9:
#             print(f"{v.VarName} = {v.X}")
# else:
#     print("Final MIP not optimal. Status:", final_mip.status)

# # Optionally write final models
# final_lp.write("PassengerMixFlow_final_lp.lp")
# final_mip.write("PassengerMixFlow_final_mip.lp")