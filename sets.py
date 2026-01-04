
import pandas as pd
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent

# Single global toggle to select data/schema
# True  -> Lecture example schema + AE4423_PMF_Exercise_Input.xlsx
# False -> Assignment schema + Group_11.xlsx
USE_LECTURE_SCHEMA = False

def _current_schema() -> str:
    return "lecture" if USE_LECTURE_SCHEMA else "assignment"

# Choose Excel input based on the global toggle above
data_file = SCRIPT_DIR / ("AE4423_PMF_Exercise_Input.xlsx" if USE_LECTURE_SCHEMA else "Group_11.xlsx")

xls = pd.ExcelFile(data_file)  # ExcelFile is not subscriptable; pass it to pd.read_excel
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

# def get_flights():
#     df = pd.read_excel(xls, sheet_name="Flights")  # header row assumed
#     flights = []
#     for r in df.itertuples(index=False):  # position-based to avoid column name mismatches
#         flight = Flights(
#             flight_id=r[0],
#             origin=r[1],
#             destination=r[2],
#             departure=r[3],
#             ready=r[4],
#             capacity=r[5]
#         )
#         flights.append(flight)
#     return flights

# def get_itineraries():
#     df = pd.read_excel(xls, sheet_name="Itineraries")
#     itineraries = []
#     for r in df.itertuples(index=False):
#         itinerary = Itineraries(
#             itinerary_id=r[0],
#             origin=r[1],
#             destination=r[2],
#             flight1=r[3],
#             flight2=r[4],
#             price=r[5],
#             demand=r[6]
#         )
#         itineraries.append(itinerary)

#     # Add a dummy itinerary with fare 0
#     dummy_itinerary = Itineraries(
#         itinerary_id=382,
#         origin=None,
#         destination=None,
#         flight1=None,
#         flight2=None,
#         price=0,
#         demand=0
#     ) 
#     itineraries.append(dummy_itinerary)

#     return itineraries


# def get_recapture():
#     df = pd.read_excel(xls, sheet_name="Recapture")
#     recaptures = []
#     for r in df.itertuples(index=False):
#         recapture = Recapture(
#             from_itinerary=r[0],
#             to_itinerary=r[1],
#             recapture_rate=r[2]
#         )
#         recaptures.append(recapture)
#     # Add recapture to dummy itinerary for all p
#     for itin in get_itineraries():
#         recapture = Recapture(
#         from_itinerary=itin.itinerary_id,
#         to_itinerary=382,
#         recapture_rate=1.0)
    
#         recaptures.append(recapture)
#     return recaptures


# Lecture example usage:


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

def get_itineraries(schema: str | None = None):
    if schema is None:
        schema = _current_schema()
    df = pd.read_excel(xls, sheet_name="Itineraries")
    itineraries = []
    for r in df.itertuples(index=False):
        if schema == "lecture":
            itinerary = Itineraries(
                r[0],   # itinerary_id
                r[1],   # origin
                r[6],   # flight1
                r[7],   # flight2
                r[2],   # destination
                r[3],   # demand
                r[4]    # price
            )
        else:
            itinerary = Itineraries(
                r[0],   # itinerary_id
                r[1],   # origin
                r[3],   # flight1
                r[4],   # flight2
                r[2],   # destination
                r[6],   # demand
                r[5]    # price
            )
        itineraries.append(itinerary)

    dummy_itinerary = Itineraries(
        382,
        None,
        None,
        None,
        None,
        0,
        0
    )
    itineraries.append(dummy_itinerary)

    return itineraries


def get_recapture(schema: str | None = None):
    if schema is None:
        schema = _current_schema()
    df = pd.read_excel(xls, sheet_name="Recapture")
    recaptures = []
    for r in df.itertuples(index=False):
        if schema == "lecture":
            recapture = Recapture(
                from_itinerary=r[1],
                to_itinerary=r[2],
                recapture_rate=r[3]
            )
        else:
            recapture = Recapture(
                from_itinerary=r[0],
                to_itinerary=r[1],
                recapture_rate=r[2]
            )
        recaptures.append(recapture)

    for itin in get_itineraries(schema):
        recapture = Recapture(itin.itinerary_id, itin.itinerary_id, 1.0)
        recaptures.append(recapture)
        recapture = Recapture(itin.itinerary_id, 382, 1.0)
        recaptures.append(recapture)
        recapture = Recapture(382, itin.itinerary_id, 0.0)
        recaptures.append(recapture)
    return recaptures