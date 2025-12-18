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
# (used later to determine which pairs have valid recapture rates)
allowed_pairs = set()
for p_itin in itins:
    for r_itin in itins:
        p_id, r_id = p_itin.itinerary_id, r_itin.itinerary_id
        if p_id == r_id or (p_id, r_id) in recapture_map:
            allowed_pairs.add((p_id, r_id))

# Variables: create for ALL P x P combinations (|P|^2 variables)
# This is for instructive purposes to show full variable space
x_vars = {}
for p_itin in itins:
    for r_itin in itins:
        p_id, r_id = p_itin.itinerary_id, r_itin.itinerary_id
        var_name = f"x_{p_id}_{r_id}"
        x_vars[(p_id, r_id)] = model_expensive.addVar(vtype=GRB.INTEGER, name=var_name, lb=0)

model_expensive.update()

# Print variable and constraint counts for verification
num_itins = len(itins)
num_flights = len(flights)
print(f"\n--- Model Size (Instructive) ---")
print(f"Number of itineraries |P|: {num_itins}")
print(f"Number of flights: {num_flights}")
print(f"Decision variables created: {len(x_vars)} (should be |P|^2 = {num_itins}^2 = {num_itins**2})")
print(f"Expected constraints: {num_flights} (capacity) + {num_itins} (demand) = {num_flights + num_itins}")

# C1: Capacity Constraints (|flights| constraints)
# sum_{p in P} sum_{r in P} delta_i^r * x_p^r <= CAP_i
for flight in flights:
    flight_id = flight.flight_id
    constr_expr = gp.LinExpr()

    for p_itin in itins:
        p_id = p_itin.itinerary_id
        for r_itin in itins:
            r_id = r_itin.itinerary_id
            # delta_i^r = 1 if flight is in itinerary r (all variables included)
            if flight_id in itin_flights.get(r_id, []):
                constr_expr += x_vars[(p_id, r_id)]
    
    model_expensive.addConstr(constr_expr <= flight.capacity, name=f"C1_Capacity_{flight_id}")

# C2: Demand Constraints (|P| constraints)
# sum_{r in P} (x_p^r / b_p^r) <= demand_p
# Note: Only include terms where b_p^r is defined (p==r or valid recapture)
# Variables without valid recapture have 0 coefficient (excluded from sum)
for p_itin in itins:
    p_id = p_itin.itinerary_id
    constr_expr = gp.LinExpr()
    
    for r_itin in itins:
        r_id = r_itin.itinerary_id
        x_var = x_vars[(p_id, r_id)]
        # Only add term if this is a valid recapture pair
        if (p_id, r_id) in allowed_pairs:
            if p_id == r_id:
                b_pr = 1.0
            else:
                b_pr = recapture_map[(p_id, r_id)]
            constr_expr += x_var / b_pr
        # else: coefficient is 0 (variable not included in this constraint)
    
    model_expensive.addConstr(constr_expr <= p_itin.demand, name=f"C2_Demand_{p_id}")

# Objective: maximize revenue = sum_{p} sum_{r} fare_r * x_p^r
# Only valid recapture pairs contribute to revenue (others have 0 coefficient)
obj_expr = gp.LinExpr()
for p_itin in itins:
    p_id = p_itin.itinerary_id
    for r_itin in itins:
        r_id = r_itin.itinerary_id
        if (p_id, r_id) in allowed_pairs:
            # revenue depends on the received itinerary r
            obj_expr += fare_map[r_id] * x_vars[(p_id, r_id)]
        # else: coefficient is 0 (no revenue for invalid recapture)

model_expensive.setObjective(obj_expr, GRB.MAXIMIZE)

model_expensive.write("passenger_mix_flow_expensive.lp")
model_expensive.optimize()

# Print actual constraint count for verification
print(f"\nActual constraints in model: {model_expensive.NumConstrs}")
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
