from sets import get_flights, get_itineraries, get_recapture
import gurobipy as gp
from gurobipy import GRB

flights = get_flights()
itins = get_itineraries()
recaps = get_recapture()



model_expensive = gp.Model("Passenger_Mix_Flow_Expensive")

# number of passangers from itin p that will travel on itinerary r
for itin in itins:
    for itin2 in itins:
        var_name = f"t_{itin.itinerary_id}_{itin2.itinerary_id}"
        model_expensive.addVar(vtype=GRB.INTEGER, name=var_name, lb=0)

# constraints

for flight in flights:

    flight_id = flight.flight_id
    constr_expr = gp.LinExpr()

    for itin in itins:
        for itin2 in itins:
            # check if flight is in itinerary p
            if flight_id == itin.flight1 or flight_id == itin.flight2:
                var_name = f"t_{itin.itinerary_id}_{itin2.itinerary_id}"
                t_var = model_expensive.getVarByName(var_name)
                constr_expr += t_var

            # check if flight is in itinerary r
            if flight_id == itin2.flight1 or flight_id == itin2.flight2:
                # get recapture rate
                b_rp = next((r.recapture_rate for r in recaps if r.from_itinerary == itin2.itinerary_id and r.to_itinerary == itin.itinerary_id), 0)
                var_name = f"t_{itin2.itinerary_id}_{itin.itinerary_id}"
                t_var = model_expensive.getVarByName(var_name)
                constr_expr -= b_rp * t_var

    # RHS
    Q_i = sum(itin.demand for itin in itins if flight_id == itin.flight1 or flight_id == itin.flight2)
    rhs = Q_i - flight .capacity

    model_expensive.addConstr(constr_expr >= rhs, name=f"C1_Capacity_{flight_id}")