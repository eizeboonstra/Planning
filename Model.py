import numpy as np
import scipy as sp
import pandas as pd

Demand=pd.read_csv('DemandMatrix.csv',header=None)
print(Demand[0][1])

pop = pd.read_csv('pop.csv') # "City", "2021", "2024"

print(pop['2021'][0])

GDP = pd.read_csv('GDP.csv') # "City", "2021", "2024"
print(GDP['2021'][0])