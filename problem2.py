import gurobipy as gp
from gurobipy import GRB
from sets import get_flights, get_itineraries, get_recapture, Recapture
import time
import matplotlib.pyplot as plt
script_start = time.perf_counter()





# Passenger Mix Flow problem
model = gp.Model("Passenger_Mix_Flow")

def _norm_id(x):
    try:
        if isinstance(x, float) and x.is_integer():
            return str(int(x))
        if isinstance(x, (int,)):
            return str(x)
        # Try to parse numeric strings like '302.0'
        xi = float(x)
        return str(int(xi)) if xi.is_integer() else str(x)
    except Exception:
        return str(x)

#=============================================================================================================================

flights = get_flights()
itins = get_itineraries()  # uses global toggle in sets.py
recaps = get_recapture()   # uses global toggle in sets.py
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
            b_pr = recapture_map.get((p_id, r_id), 0)  # recapture rate from p to r

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
            b_pr = recapture_map.get((p_id, r_id), 0) # recapture rate from r to p

            # Check if flight_id is in itinerary r
            if flight_id in itin_flights.get(r_id, []):
                var_name = f"t_{p_id}_{r_id}"
                t_var = model.getVarByName(var_name)
                if t_var:
                    constr_expr -= b_pr * t_var

        # RHS: use original difference Q_i - CAP_i (do NOT clamp to 0)
        rhs = flight_demand_Q.get(flight_id, 0) - flight.capacity



        # Always add the constraint (if constr_expr is zero this becomes 0 >= rhs)
        model.addConstr(constr_expr >= rhs, name=f"C1_Capacity_{_norm_id(flight_id)}")

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
        model.addConstr(constr_expr <= rhs, name=f"C2_Demand_{_norm_id(p_id)}")

    # different demand formulation

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

#model.optimize()

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
    #lp_value = model.objVal

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
        b_pr = recapture_map.get((p_id, r_id), 0)  # b(p→r) for objective

        flights_in_p = set(itin_flights.get(p_id, []))
        flights_in_r = set(itin_flights.get(r_id, []))
        all_flights = flights_in_p | flights_in_r
        
        # C1 contribution - must match build_model_constraints logic!
        # For variable t_{p}_{r}:
        #   Term 1: +1 if flight in p

        dual_sum_c1 = 0
        for flight_id in all_flights:
            constr = model.getConstrByName(f"C1_Capacity_{_norm_id(flight_id)}")
            if constr is not None:
                pi_i = constr.Pi
                delta_p = 1 if flight_id in flights_in_p else 0
                delta_r = 1 if flight_id in flights_in_r else 0
                b_pr = recapture_map.get((p_id,r_id), 0)  
                coeff = delta_p - delta_r * b_pr  # Match constraint!
                dual_sum_c1 += coeff * pi_i

        # C2 contribution: sigma_p
        demand_constr = model.getConstrByName(f"C2_Demand_{_norm_id(p_id)}")
        sigma_p = demand_constr.Pi if demand_constr is not None else 0

        # Objective coefficient uses b_pr 
        cost_coeff = fare_p - b_pr * fare_r
        
        # Reduced cost = c_j - (a_j^T * pi)
        slackness_value = cost_coeff - dual_sum_c1 - sigma_p
        slackness[(p_id, r_id)] = slackness_value

    min_slackness = min(slackness.values())
    
    negative_rc_count = sum(1 for v in slackness.values() if v < -1e-6)
    print(f"  Columns with negative reduced cost: {negative_rc_count}")

    # remove columns with negative slackness and add them to master
    columns_to_remove = []
    columns_added_count = 0
    for key, value in slackness.items():
        if value < 0:
            p_id, r_id = key
            col_to_move = next((c for c in columns if c.from_itinerary == p_id and c.to_itinerary == r_id), None)
            if col_to_move:
                columns_to_remove.append(col_to_move)
                master.append(col_to_move)
                columns_added_count += 1
                # Add the variable to the model for the new column
                var_name = f"t_{p_id}_{r_id}"
                if model.getVarByName(var_name) is None:
                    model.addVar(vtype=GRB.CONTINUOUS, name=var_name, lb=0)
    
    # Remove columns after iteration to avoid modifying list during iteration
    for col in columns_to_remove:
        columns.remove(col)
    
    model.update()

    return min_slackness, master, columns, columns_added_count

def calculate_slackness_reformulation(itins, recaps, model, columns, master):
    slackness = {}
    fare_map = {itin.itinerary_id: itin.price for itin in itins}
    recapture_map = {(r.from_itinerary, r.to_itinerary): r.recapture_rate for r in recaps}
    itin_flights = {itin.itinerary_id: [itin.flight1, itin.flight2] for itin in itins}

    for column in columns:
        p_id = column.from_itinerary
        r_id = column.to_itinerary

        fare_p = fare_map.get(p_id, 0)
        fare_r = fare_map.get(r_id, 0)
        b_pr = recapture_map.get((p_id, r_id), 0)

        flights_in_p = itin_flights.get(p_id, []) or []
        flights_in_r = itin_flights.get(r_id, []) or []

        sum_pi_p = 0
        for flight_id in flights_in_p:
            constr = model.getConstrByName(f"C1_Capacity_{_norm_id(flight_id)}")
            if constr is not None:
                sum_pi_p += constr.Pi

        sum_pi_r = 0
        for flight_id in flights_in_r:
            constr = model.getConstrByName(f"C1_Capacity_{_norm_id(flight_id)}")
            if constr is not None:
                sum_pi_r += constr.Pi

        demand_constr = model.getConstrByName(f"C2_Demand_{_norm_id(p_id)}")
        sigma_p = demand_constr.Pi if demand_constr is not None else 0

        modified_fare_p = fare_p - sum_pi_p
        modified_fare_r = fare_r - sum_pi_r

        reduced_cost = modified_fare_p - b_pr * modified_fare_r - sigma_p
        slackness[(p_id, r_id)] = reduced_cost

    min_slackness = min(slackness.values()) if slackness else 0

    columns_to_remove = []
    added_var_names = []
    for key, value in slackness.items():
        if value < 0:
            p_id, r_id = key
            col_to_move = next((c for c in columns if c.from_itinerary == p_id and c.to_itinerary == r_id), None)
            if col_to_move:
                columns_to_remove.append(col_to_move)
                master.append(col_to_move)
                var_name = f"t_{p_id}_{r_id}"
                if model.getVarByName(var_name) is None:
                    model.addVar(vtype=GRB.CONTINUOUS, name=var_name, lb=0)
                    added_var_names.append(var_name)

    for col in columns_to_remove:
        columns.remove(col)

    model.update()

    return min_slackness, master, columns, added_var_names



iteration = 0
objective_history = []  # Track objective values per iteration
lp_iteration_times = []  # Track solve time per LP iteration
lp_total_time = 0.0
columns_added_per_iteration = []  # Track columns added per iteration

while True:
    # Remove old constraints before rebuilding
    for c in model.getConstrs():
        model.remove(c)
    model.update()
    
    build_model_constraints(model, flights, itins, master)
    set_objective_function(model, itins, master)
    iter_start = time.perf_counter()
    model.optimize()
    iter_time = time.perf_counter() - iter_start
    lp_iteration_times.append(iter_time)
    lp_total_time += iter_time

    
    # Store objective value for this iteration
    if model.status == GRB.OPTIMAL:
        objective_history.append(model.objVal)
    
    if not columns:  # No more columns to check
        break
        
    min_slackness, master, columns, cols_added = calculate_slackness(itins, recaps, model, columns, master)
    columns_added_per_iteration.append(cols_added)
    iteration += 1
    print(f"Iteration {iteration}: Minimum Slackness = {min_slackness}, Columns Added = {cols_added}")
    
    # if added_vars:
    #     print("  Added DVs:", ", ".join(added_vars))
    # else:
    #     print("  Added DVs: none")
    
    if min_slackness >= 0:
        break

# Write the LP file for the column generation (LP relaxation before integer conversion)
model.write("ColumnGeneration_LP.lp")
print(f"\nLP file written to: ColumnGeneration_LP.lp")

# Print first 5 nonzero duals for capacity constraints (last LP relaxation)
if model.status == GRB.OPTIMAL:
    print("\nFirst 5 nonzero duals for capacity constraints (last LP relaxation):")
    count = 0
    for flight in flights:
        constr = model.getConstrByName(f"C1_Capacity_{_norm_id(flight.flight_id)}")
        if constr is not None and abs(constr.Pi) > 1e-6:
            print(f"Flight {flight.flight_id}: Dual value (Pi) = {constr.Pi:.4f}")
            count += 1
            if count == 5:
                break




# Optimize using integer variables and additional columns
for r in master:
    p_id = r.from_itinerary
    r_id = r.to_itinerary
    var_name = f"t_{p_id}_{r_id}"
    var = model.getVarByName(var_name)
    var.vtype = GRB.INTEGER
model.update()
mip_start = time.perf_counter()
model.optimize()
mip_time = time.perf_counter() - mip_start

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

# Print objective value history
print("\n--- Objective Value per Iteration ---")
for i, obj_val in enumerate(objective_history):
    print(f"  Iteration {i}: {obj_val:.2f}")

print("\n--- Computational Time ---")
print(f"Total LP solve time (column generation): {lp_total_time:.3f} seconds")
print(f"Final MIP solve time: {mip_time:.3f} seconds")
print(f"Total computational time: {lp_total_time + mip_time:.3f} seconds")
print(f"End-to-end script runtime: {time.perf_counter() - script_start:.3f} seconds")

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


#print(f"\nDifference between LP and column generation MIP objective values: {abs(lp_value - model.objVal)}")
#print("All columns active and integer", lp_value)

# --- Comparison with expensive_model.py ---
total_potential_revenue = sum(itin.demand * itin.price for itin in itins)
print(f"\n--- Model Comparison ---")
print(f"Total Potential Revenue: ${total_potential_revenue:,.2f}")
print(f"Lost Revenue (this model): ${model.objVal:,.2f}")
print(f"Implied Actual Revenue: ${total_potential_revenue - model.objVal:,.2f}")
print(f"\nTo verify: expensive_model.py objective should equal: ${total_potential_revenue - model.objVal:,.2f}")

# # =============================================================
# # PLOTTING: Objective Trajectory & Columns Added per Iteration
# # =============================================================

# fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# # Plot 1: Objective Trajectory
# ax1 = axes[0]
# ax1.plot(range(len(objective_history)), objective_history, marker='o', linewidth=2, markersize=6, color='blue')
# ax1.set_xlabel('Iteration', fontsize=12)
# ax1.set_ylabel('Objective Value (Lost Revenue)', fontsize=12)
# ax1.set_title('Column Generation: Objective Trajectory', fontsize=14)
# ax1.grid(True, alpha=0.3)
# ax1.ticklabel_format(style='plain', axis='y')

# # Plot 2: Columns Added per Iteration
# ax2 = axes[1]
# ax2.bar(range(1, len(columns_added_per_iteration) + 1), columns_added_per_iteration, color='green', alpha=0.7)
# ax2.set_xlabel('Iteration', fontsize=12)
# ax2.set_ylabel('Columns Added', fontsize=12)
# ax2.set_title('Columns Added per Iteration', fontsize=14)
# ax2.grid(True, alpha=0.3, axis='y')

# plt.tight_layout()
# plt.show()

# =============================================================
# KPI: REVENUE OPPORTUNITY - Where to Invest for More Revenue
# =============================================================

if model.status == GRB.OPTIMAL:
    print("\n" + "="*70)
    print("REVENUE OPPORTUNITY: Where to Invest for More Revenue")
    print("="*70)
    print("\nOD pairs ranked by lost revenue (spilled passengers × average fare)")
    print("Adding capacity on these routes would recapture this revenue.\n")
    
    # Calculate lost revenue per OD pair
    od_opportunity = {}
    
    for itin in itins:
        if itin.itinerary_id == 382:
            continue
        
        od = (itin.origin, itin.destination)
        if od not in od_opportunity:
            od_opportunity[od] = {'spilled_pax': 0, 'lost_revenue': 0, 'avg_fare': 0, 'total_fare': 0, 'count': 0}
        
        # Get spilled passengers for this itinerary
        spilled_var = model.getVarByName(f"t_{itin.itinerary_id}_382")
        spilled = spilled_var.X if spilled_var and abs(spilled_var.X) > 1e-6 else 0
        
        if spilled > 0:
            od_opportunity[od]['spilled_pax'] += spilled
            od_opportunity[od]['lost_revenue'] += spilled * itin.price
            od_opportunity[od]['total_fare'] += itin.price
            od_opportunity[od]['count'] += 1
    
    # Calculate average fare for ODs with spill
    for od in od_opportunity:
        if od_opportunity[od]['count'] > 0:
            od_opportunity[od]['avg_fare'] = od_opportunity[od]['total_fare'] / od_opportunity[od]['count']
    
    # Filter to only ODs with lost revenue and sort
    opportunities = [(od, data) for od, data in od_opportunity.items() if data['lost_revenue'] > 0]
    opportunities_sorted = sorted(opportunities, key=lambda x: x[1]['lost_revenue'], reverse=True)
    
    print(f"{'Rank':<5} {'Origin':<8} {'Dest':<8} {'Spilled':>10} {'Avg Fare':>12} {'Lost Revenue':>14}")
    print("-" * 60)
    
    for rank, (od, data) in enumerate(opportunities_sorted[:15], 1):
        print(f"{rank:<5} {od[0]:<8} {od[1]:<8} {data['spilled_pax']:>10.0f} ${data['avg_fare']:>10.2f} ${data['lost_revenue']:>12,.0f}")
    
    total_lost = sum(data['lost_revenue'] for _, data in opportunities_sorted)
    total_spilled = sum(data['spilled_pax'] for _, data in opportunities_sorted)
    
    print("-" * 60)
    print(f"{'TOTAL':<22} {total_spilled:>10.0f} {' ':>12} ${total_lost:>12,.0f}")
    print(f"\nInvesting in the top 5 routes could recover: ${sum(data['lost_revenue'] for _, data in opportunities_sorted[:5]):,.0f}")


