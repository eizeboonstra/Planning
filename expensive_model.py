from sets import get_flights, get_itineraries, get_recapture
import gurobipy as gp
from gurobipy import GRB

flights = get_flights()
# Filter out dummy itinerary (id 382)
itins = [itin for itin in get_itineraries() if itin.itinerary_id != 382]
# Filter out recaptures involving dummy itinerary
recaps = [r for r in get_recapture() if r.from_itinerary != 382 and r.to_itinerary != 382]



model_expensive = gp.Model("Passenger_Mix_Flow_Expensive")

# number of passangers from itin p that will travel on itinerary r
for itin in itins:
    for itin2 in itins:
        var_name = f"x_{itin.itinerary_id}_{itin2.itinerary_id}"
        model_expensive.addVar(vtype=GRB.INTEGER, name=var_name, lb=0)

# Update the model to make new variables available for lookup by name
model_expensive.update()

# constraints

for flight in flights:

    flight_id = flight.flight_id
    constr_expr = gp.LinExpr()

    for itin in itins:
        for itin2 in itins:


            # check if flight is in itinerary r
            if flight_id == itin2.flight1 or flight_id == itin2.flight2:
                var_name = f"x_{itin2.itinerary_id}_{itin.itinerary_id}"
                x_var = model_expensive.getVarByName(var_name)
                constr_expr +=  x_var
    
    rhs = flight.capacity
    model_expensive.addConstr(constr_expr <= rhs, name=f"C1_Capacity_{flight_id}_upper")


# C2 number of passengers is lower than demand
for itin in itins:
    constr_expr = gp.LinExpr()
    for itin2 in itins:
        var_name = f"x_{itin.itinerary_id}_{itin2.itinerary_id}"
        x_var = model_expensive.getVarByName(var_name)
        # Get recapture rate b_p^r for this pair
        b_pr = 1.0  # default if no recapture exists
        for recap in recaps:
            if recap.from_itinerary == itin.itinerary_id and recap.to_itinerary == itin2.itinerary_id:
                b_pr = recap.recapture_rate
                break
        constr_expr += x_var / b_pr
    rhs = itin.demand
    model_expensive.addConstr(constr_expr <= rhs, name=f"C2_Demand_{itin.itinerary_id}_upper")


# Objective: maximize revenue
obj_expr = gp.LinExpr()
for itin in itins:
    for itin2 in itins:
        var_name = f"x_{itin.itinerary_id}_{itin2.itinerary_id}"
        x_var = model_expensive.getVarByName(var_name)
        price = itin2.price
        obj_expr += price * x_var
model_expensive.setObjective(obj_expr, GRB.MAXIMIZE)

# Write the model to an LP file
model_expensive.write("passenger_mix_flow_expensive.lp")

model_expensive.optimize()

# --- Results Summary ---
if model_expensive.status == GRB.OPTIMAL:
    print("\n--------------------------------")
    print("--- Optimization Successful! ---")
    print("--------------------------------")
    print(f"\nOptimal Total Revenue: ${model_expensive.objVal:,.2f}")
    
    print("\n--- Passenger Assignments (x_p_r) ---")
    for v in model_expensive.getVars():
        if v.x > 0.5:  # Print non-zero assignments
            print(f"  {v.varName}: {v.x}")
    print("\nLP model file created: passenger_mix_flow_expensive.lp")

elif model_expensive.status == GRB.INFEASIBLE:
    print("\nModel is infeasible.")
    print("Computing Irreducible Inconsistent Subsystem (IIS)...")
    model_expensive.computeIIS()
    model_expensive.write("passenger_mix_infeasible.ilp")
    print("IIS written to passenger_mix_infeasible.ilp for analysis.")
else:
    print(f"\nOptimization finished with status: {model_expensive.status}")
