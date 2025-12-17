from sets import get_flights, get_itineraries, get_recapture
import gurobipy as gp
from gurobipy import GRB
import csv

"""Schema is now controlled centrally in sets.py via USE_LECTURE_SCHEMA."""

flights = get_flights()

# Filter out dummy itinerary (id 382)
itins = [itin for itin in get_itineraries() if itin.itinerary_id != 382]
# Filter out recaptures involving dummy itinerary
recaps = [r for r in get_recapture() if r.from_itinerary != 382 and r.to_itinerary != 382]

# Build lookup maps
recapture_map = {(r.from_itinerary, r.to_itinerary): r.recapture_rate for r in recaps}
itin_flights = {itin.itinerary_id: [itin.flight1, itin.flight2] for itin in itins}
fare_map = {itin.itinerary_id: itin.price for itin in itins}
demand_map = {itin.itinerary_id: itin.demand for itin in itins}

model_expensive = gp.Model("Passenger_Mix_Flow_Expensive")

# Write Gurobi solver log to a file in the workspace
model_expensive.setParam('LogFile', 'log_file')

# Build allowed recapture pairs once: include only p==r and valid recaptures
allowed_pairs = set()
for p_itin in itins:
    for r_itin in itins:
        p_id, r_id = p_itin.itinerary_id, r_itin.itinerary_id
        if p_id == r_id or (p_id, r_id) in recapture_map:
            allowed_pairs.add((p_id, r_id))

# Variables: create only for allowed pairs to avoid extra rows/constraints
x_vars = {}
for (p_id, r_id) in allowed_pairs:
    var_name = f"x_{p_id}_{r_id}"
    x_vars[(p_id, r_id)] = model_expensive.addVar(vtype=GRB.INTEGER, name=var_name, lb=0)

model_expensive.update()

# C1: Capacity Constraints
# sum_{p in P} sum_{r in P} delta_i^r * x_p^r <= CAP_i
for flight in flights:
    flight_id = flight.flight_id
    constr_expr = gp.LinExpr()

    for p_itin in itins:
        p_id = p_itin.itinerary_id
        for r_itin in itins:
            r_id = r_itin.itinerary_id
            # delta_i^r = 1 if flight is in itinerary r
            if (p_id, r_id) in x_vars and flight_id in itin_flights.get(r_id, []):
                constr_expr += x_vars[(p_id, r_id)]
    
    model_expensive.addConstr(constr_expr <= flight.capacity, name=f"C1_Capacity_{flight_id}")

# C2: Demand Constraints
# sum_{r in P_p} (x_p^r / b_p^r) <= D_p
for p_itin in itins:
    p_id = p_itin.itinerary_id
    constr_expr = gp.LinExpr()
    
    for r_itin in itins:
        r_id = r_itin.itinerary_id
        if (p_id, r_id) not in x_vars:
            continue
        x_var = x_vars[(p_id, r_id)]
        # Demand constraint uses b_p^r; p==r has b=1, else use recapture rate
        if p_id == r_id:
            b_pr = 1.0
        else:
            b_pr = recapture_map[(p_id, r_id)]
        constr_expr += x_var / b_pr
    
    model_expensive.addConstr(constr_expr <= p_itin.demand, name=f"C2_Demand_{p_id}")

# Objective: maximize revenue = sum_{p} sum_{r} fare_r * x_p^r
obj_expr = gp.LinExpr()
for (p_id, r_id) in x_vars:
    # revenue depends on the received itinerary r
    obj_expr += fare_map[r_id] * x_vars[(p_id, r_id)]

model_expensive.setObjective(obj_expr, GRB.MAXIMIZE)

model_expensive.write("passenger_mix_flow_expensive.lp")
model_expensive.optimize()

print("Solver log written to: log_file")

# --- Results Summary ---
if model_expensive.status == GRB.OPTIMAL:
    print("\n--------------------------------")
    print("--- Optimization Successful! ---")
    print("--------------------------------")
    print(f"\nOptimal Total Revenue: ${model_expensive.objVal:,.2f}")
    
    print("\n--- Passenger Assignments (x_p_r > 0) ---")
    for v in model_expensive.getVars():
        if v.x > 0.5:
            print(f"  {v.varName}: {v.x}")
    print("\nLP model file created: passenger_mix_flow_expensive.lp")

elif model_expensive.status == GRB.INFEASIBLE:
    print("\nModel is infeasible.")
    model_expensive.computeIIS()
    model_expensive.write("passenger_mix_infeasible.ilp")
    print("IIS written to passenger_mix_infeasible.ilp")
else:
    print(f"\nOptimization finished with status: {model_expensive.status}")



# --- Comparison with problem2.py ---
total_potential_revenue = sum(itin.demand * itin.price for itin in itins)
print(f"\n--- Model Comparison ---")
print(f"Total Potential Revenue (all demand satisfied): ${total_potential_revenue:,.2f}")
print(f"Actual Revenue (expensive model): ${model_expensive.objVal:,.2f}")
print(f"Lost Revenue (expensive model): ${total_potential_revenue - model_expensive.objVal:,.2f}")
print(f"\nTo verify: problem2.py objective should equal: ${total_potential_revenue - model_expensive.objVal:,.2f}")
