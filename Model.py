import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Demand=pd.read_csv('DemandMatrix.csv',header=None)
print(Demand[0][1])
DemandLn = np.log(Demand)
np.fill_diagonal(DemandLn.values, 0)

pop = pd.read_csv('pop.csv') # "City", "2021", "2024"
print(pop['2021'][0])

GDP = pd.read_csv('GDP.csv') # "City", "2021", "2024"
print(GDP['2021'][0])

Cords = pd.read_csv('Coordinates.csv') # "City", "Latitude (deg)", "Longitude (deg)"
print(Cords['Latitude (deg)'][0])

def distance(lat1, lon1, lat2, lon2):
    R_E = 6371*1000  # Earth radius in meters
    distance = 2 * R_E * np.arcsin(np.sqrt(np.sin((lat2 - lat1) * np.pi / 360) ** 2 +
                             np.cos(lat1 * np.pi / 180) * np.cos(lat2 * np.pi / 180) *
                             np.sin((lon2 - lon1) * np.pi / 360) ** 2))
    return distance

def model(popi, popy, GDPi, GDPy, d, f, k, b1, b2, b3):
    Dln = np.log(k)+b1*np.log(popi)+b1*np.log(popy)+b2*np.log(GDPi)+b2*np.log(GDPy)-b3*np.log(d)-b3*np.log(f)
    return Dln


f = 1.42

pairs = []
for i in range(20):
    for j in range(20):
        if i == j:
            continue
        
        D_ij = DemandLn.iloc[i, j]

        pop_i = pop['2021'][i]
        pop_j = pop['2021'][j]

        GDP_i = GDP['2021'][i]
        GDP_j = GDP['2021'][j]

        d_ij = distance(Cords['Latitude (deg)'][i],
                        Cords['Longitude (deg)'][i],
                        Cords['Latitude (deg)'][j],
                        Cords['Longitude (deg)'][j])

        pairs.append([
            D_ij,
            np.log(pop_i) + np.log(pop_j),
            np.log(GDP_i) + np.log(GDP_j),
            np.log(d_ij) + np.log(f)
        ])

pairs = np.array(pairs)

# Separate dependent and independent variables
Y = pairs[:, 0]                       # log(Demand)
X = pairs[:, 1:]                      # log(pop_i pop_j), log(GDP_i GDP_j), log(d*f)

# Add constant column for ln(k)
X = np.column_stack([np.ones(len(X)), X])

# Solve least squares:  beta = [ ln(k), b1, b2, -b3 ]
beta, *_ = np.linalg.lstsq(X, Y, rcond=None)

ln_k, b1, b2, neg_b3 = beta
b3 = -neg_b3
k = np.exp(ln_k)

print("k  =", k)
print("b1 =", b1)
print("b2 =", b2)
print("b3 =", b3)

# Estimated demand matrix
Demand_est = np.zeros_like(Demand, dtype=float)

for i in range(20):
    for j in range(20):
        if i == j:
            continue
        
        pop_i = pop['2021'][i]
        pop_j = pop['2021'][j]

        GDP_i = GDP['2021'][i]
        GDP_j = GDP['2021'][j]

        d_ij = distance(Cords['Latitude (deg)'][i],
                        Cords['Longitude (deg)'][i],
                        Cords['Latitude (deg)'][j],
                        Cords['Longitude (deg)'][j])

        Demand_est[i, j] = (
            k
            * (pop_i**b1) * (pop_j**b1)
            * (GDP_i**b2) * (GDP_j**b2)
            * ((d_ij * f)**(-b3))
        )

# Flatten both matrices (exclude diagonal)
actual = []
pred   = []

for i in range(20):
    for j in range(20):
        if i == j:
            continue
        actual.append(Demand.iloc[i, j])
        pred.append(Demand_est[i, j])

actual = np.array(actual)
pred   = np.array(pred)

cities = pop["City"]
Demand_est_df = pd.DataFrame(np.zeros((20, 20)), index=cities, columns=cities)

for i in range(20):
    for j in range(20):
        if i == j:
            Demand_est_df.iloc[i, j] = 0
            continue

        pop_i = pop["2021"][i]
        pop_j = pop["2021"][j]

        GDP_i = GDP["2021"][i]
        GDP_j = GDP["2021"][j]

        d_ij = distance(Cords["Latitude (deg)"][i],
                        Cords["Longitude (deg)"][i],
                        Cords["Latitude (deg)"][j],
                        Cords["Longitude (deg)"][j])

        Demand_est_df.iloc[i, j] = (
            k
            * (pop_i ** b1) * (pop_j ** b1)
            * (GDP_i ** b2) * (GDP_j ** b2)
            * ((d_ij * 1.42) ** (-b3))
        )


# # Scatter plot
# plt.figure(figsize=(6,6))
# plt.scatter(actual, pred, s=10)
# plt.xlabel("Actual Demand")
# plt.ylabel("Estimated Demand")
# plt.title("Gravity Model: Actual vs Estimated Demand")

# # 45-degree reference line
# lims = [0, max(actual.max(), pred.max())]
# plt.plot(lims, lims)

# plt.show()


pop_array = pop['2024'].values + 2 * ( pop['2024'].values - pop['2021'].values ) / 3
GDP_array = GDP['2024'].values + 2 * ( GDP['2024'].values - GDP['2021'].values ) / 3

pop_updated = pd.DataFrame({
    'City': pop['City'],
    '2026': pop_array
})
GDP_updated = pd.DataFrame({
    'City': GDP['City'],
    '2026': GDP_array
})


Demand_2026_df = pd.DataFrame(np.zeros((20, 20)),
                              index=pop_updated['City'],
                              columns=pop_updated['City'])

for i in range(20):
    for j in range(20):
        if i == j:
            Demand_2026_df.iloc[i, j] = 0
            continue
        
        pop_i = pop_updated['2026'][i]
        pop_j = pop_updated['2026'][j]

        GDP_i = GDP_updated['2026'][i]
        GDP_j = GDP_updated['2026'][j]

        d_ij = distance(Cords["Latitude (deg)"][i],
                        Cords["Longitude (deg)"][i],
                        Cords["Latitude (deg)"][j],
                        Cords["Longitude (deg)"][j])

        Demand_2026_df.iloc[i, j] = (
            k
            * (pop_i ** b1) * (pop_j ** b1)
            * (GDP_i ** b2) * (GDP_j ** b2)
            * ((d_ij * f) ** (-b3))
        )

print(Demand_2026_df)