import gurobipy as gp
from gurobipy import GRB
from sets import get_flights, get_itineraries, get_recapture, Recapture





# Passenger Mix Flow problem
model = gp.Model("Passenger_Mix_Flow")

#=============================================================================================================================

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
                if t_var:
                    constr_expr += t_var

        # Term 2:
        for recap in recaps:
            p_id = recap.from_itinerary
            r_id = recap.to_itinerary
            b_rp = recapture_map.get((r_id, p_id), 0) # recapture rate from r to p

            # Check if flight_id is in itinerary r
            if flight_id in itin_flights.get(r_id, []):
                var_name = f"t_{r_id}_{p_id}"
                t_var = model.getVarByName(var_name)
                if t_var:
                    constr_expr -= b_rp * t_var

        # RHS: use original difference Q_i - CAP_i (do NOT clamp to 0)
        rhs = flight_demand_Q.get(flight_id, 0) - flight.capacity



        # Always add the constraint (if constr_expr is zero this becomes 0 >= rhs)
        model.addConstr(constr_expr >= rhs, name=f"C1_Capacity_{flight_id}")

    # --- C2: Demand Constraints ---
    for p_itin in itins:
        p_id = p_itin.itinerary_id
        constr_expr = gp.LinExpr()
        for recap in recaps:
            if recap.from_itinerary == p_id:
                var = model.getVarByName(f"t_{p_id}_{recap.to_itinerary}")
                if var:
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

        
        fare_p = fare_map.get(p_id, 0)
        fare_r = fare_map.get(r_id, 0)
        b_pr = recapture_map.get((p_id, r_id), 0)
        
        # Cost coefficient: (fare_p - b_p^r * fare_r)
        cost_coeff = fare_p - b_pr * fare_r
        
        obj_expr += cost_coeff * t_var
    
    model.setObjective(obj_expr, GRB.MINIMIZE)
    model.update()
    return model



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
    lp_value = model.objVal

    print("\n--- Decision Variables ---")
    for v in model.getVars():
        if abs(v.X) > 1e-6:  # print only nonzero variables
            print(f"{v.VarName} = {v.X}")

else:
    print("\nModel did not solve to optimality.")
    print(f"Gurobi Status Code: {model.status}")

#================================================================
# Implementation of column generation
#================================================================

model = gp.Model("Passenger_Mix_Flow_Column_Generation")


for recap in recaps:
    p_id = recap.from_itinerary
    r_id = recap.to_itinerary
    if r_id == 382:
        var_name = f"t_{p_id}_{r_id}"
        model.addVar(vtype=GRB.CONTINUOUS, name=var_name, lb=0)

model.update()




columns = []

for r in recaps:
    if r.to_itinerary != 382:
        columns.append(r)

master = []

for r in recaps:
    if r.to_itinerary == 382:
        master.append(r)

def calculate_slackness(itins, recaps, model, columns, master):
    slackness = {}
    fare_map = {itin.itinerary_id: itin.price for itin in itins}
    recapture_map = {(r.from_itinerary, r.to_itinerary): r.recapture_rate for r in recaps}
    itin_flights = {itin.itinerary_id: [itin.flight1, itin.flight2] for itin in itins}
    

    for column in columns:
        p_id = column.from_itinerary
        r_id = column.to_itinerary
        fare_p = fare_map.get(p_id, 0)
        fare_r = fare_map.get(r_id, 0)
        recapture_rate = recapture_map.get((p_id, r_id), 0)

        # dual for capacity constraint of each flight in itinerary p
        dual_sum = 0
        for flight_id in itin_flights.get(p_id, []):
            constr = model.getConstrByName(f"C1_Capacity_{flight_id}")
            if constr is not None:
                dual_sum += constr.Pi
        
        # dual for capacity constraint of each flight in itinerary r
        sum_j = 0
        for flight_id in itin_flights.get(r_id, []):
            constr = model.getConstrByName(f"C1_Capacity_{flight_id}")
            if constr is not None:
                sum_j += constr.Pi

        # dual for demand constraint of itinerary p
        demand_constr = model.getConstrByName(f"C2_Demand_{p_id}")
        dual_demand = demand_constr.Pi if demand_constr is not None else 0

        slackness_value = (fare_p - dual_sum) - recapture_rate * (fare_r - sum_j) - dual_demand
        slackness[(p_id, r_id)] = slackness_value

    # find minimum slackness
    min_slackness = min(slackness.values())

    # remove columns with negative slackness and add them to master
    columns_to_remove = []
    for key, value in slackness.items():
        if value < 0:
            p_id, r_id = key
            col_to_move = next((c for c in columns if c.from_itinerary == p_id and c.to_itinerary == r_id), None)
            if col_to_move:
                columns_to_remove.append(col_to_move)
                master.append(col_to_move)
                # Add the variable to the model for the new column
                var_name = f"t_{p_id}_{r_id}"
                if model.getVarByName(var_name) is None:
                    model.addVar(vtype=GRB.CONTINUOUS, name=var_name, lb=0)
    
    # Remove columns after iteration to avoid modifying list during iteration
    for col in columns_to_remove:
        columns.remove(col)
    
    model.update()

    return min_slackness, master, columns   

iteration = 0     

while True:
    # Remove old constraints before rebuilding
    for c in model.getConstrs():
        model.remove(c)
    model.update()
    
    build_model_constraints(model, flights, itins, master)
    set_objective_function(model, itins, master)
    model.optimize()
    
    if not columns:  # No more columns to check
        break
        
    min_slackness, master, columns = calculate_slackness(itins, recaps, model, columns, master)
    iteration += 1
    print(f"Iteration {iteration}: Minimum Slackness = {min_slackness}")
    
    if min_slackness >= 0:
        break

# Write the LP file for the column generation (LP relaxation before integer conversion)
model.write("ColumnGeneration_LP.lp")
print(f"\nLP file written to: ColumnGeneration_LP.lp")

# Optimize using integer variables and additional columns
for r in master:
    p_id = r.from_itinerary
    r_id = r.to_itinerary
    var_name = f"t_{p_id}_{r_id}"
    var = model.getVarByName(var_name)
    var.vtype = GRB.INTEGER
model.update()
model.optimize()

# Write the MIP file after integer conversion
model.write("ColumnGeneration_MIP.lp")
print(f"MIP file written to: ColumnGeneration_MIP.lp")

# =============================================================
# COLUMN GENERATION RESULTS SUMMARY
# =============================================================
print("\n" + "="*60)
print("COLUMN GENERATION RESULTS SUMMARY")
print("="*60)

# Print number of iterations
print(f"Total number of iterations: {iteration}")

# Calculate and print passengers recaptured and non-zero variables
if model.status == GRB.OPTIMAL:
    print(f"Optimal objective value: {model.objVal}")
    
    total_passengers_reallocated = 0
    passengers_to_dummy = 0
    passengers_recaptured = 0
    nonzero_vars_not_dummy = 0
    nonzero_vars_list = []
    
    for v in model.getVars():
        if abs(v.X) > 1e-6:  # non-zero variables
            parts = v.VarName.split('_')
            if len(parts) >= 3:
                p_id = int(parts[1])
                r_id = int(parts[2])
                
                total_passengers_reallocated += v.X
                
                if r_id == 382:
                    passengers_to_dummy += v.X
                else:
                    passengers_recaptured += v.X
                    nonzero_vars_not_dummy += 1
                    nonzero_vars_list.append((v.VarName, v.X))
    
    print("\n--- Summary Statistics ---")
    print(f"Total passengers reallocated: {total_passengers_reallocated:.0f}")
    print(f"Passengers recaptured (to real itineraries): {passengers_recaptured:.0f}")
    print(f"Passengers spilled (to dummy itinerary 382): {passengers_to_dummy:.0f}")
    print(f"Non-zero variables NOT going to dummy: {nonzero_vars_not_dummy}")
    
    if nonzero_vars_list:
        print("\n--- Non-zero Variables (real recapture) ---")
        for var_name, var_val in nonzero_vars_list:
            print(f"  {var_name} = {var_val:.0f}")
    
else:
    print(f"\nModel did not solve to optimality. Status: {model.status}")


print(f"\nDifference between LP and column generation MIP objective values: {abs(lp_value - model.objVal)}")
print("All columns active and integer", lp_value)


