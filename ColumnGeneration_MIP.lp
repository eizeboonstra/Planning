\ Model Passenger_Mix_Flow_Column_Generation
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xa62fabcdc662b02b
Minimize
  150 t_1_382 + 170 t_2_382 + 80 t_3_382 + 70 t_4_382 + 50 t_5_382
   + 120 t_6_382 + 100 t_7_382 + 100 t_8_382 + 140 t_9_382 + 80 t_10_382
   + 150 t_11_382 + 180 t_12_382 + 100 t_13_382 + 75 t_14_382 + 80 t_15_382
   + 66 t_3_4 + 75 t_3_5 + 90 t_6_7 + 80 t_7_8 + 110 t_6_8 + 124 t_9_10
   + 150 t_12_13 + 165 t_12_11
Subject To
 C1_Capacity_102: t_1_382 + t_7_382 + t_7_8 >= 62
 C1_Capacity_301: t_10_382 + t_14_382 >= 56
 C1_Capacity_201: t_5_382 + t_7_382 + t_11_382 + t_7_8 >= 36
 C1_Capacity_104: t_2_382 + t_8_382 >= 82
 C1_Capacity_202: t_3_382 + t_10_382 + t_13_382 + t_3_4 + t_3_5 >= 117
 C1_Capacity_302: t_5_382 + t_9_382 + t_9_10 >= 82
 C1_Capacity_303: t_15_382 >= -64
 C1_Capacity_204: t_4_382 >= -8
 C1_Capacity_203: t_8_382 + t_12_382 + t_12_13 + t_12_11 >= 142
 C1_Capacity_101: t_6_382 + t_13_382 + t_6_7 + t_6_8 >= 42
 C2_Demand_1: t_1_382 <= 90
 C2_Demand_2: t_2_382 <= 120
 C2_Demand_3: t_3_382 + t_3_4 + t_3_5 <= 70
 C2_Demand_4: t_4_382 <= 100
 C2_Demand_5: t_5_382 <= 50
 C2_Demand_6: t_6_382 + t_6_7 + t_6_8 <= 120
 C2_Demand_7: t_7_382 + t_7_8 <= 80
 C2_Demand_8: t_8_382 <= 70
 C2_Demand_9: t_9_382 + t_9_10 <= 140
 C2_Demand_10: t_10_382 <= 125
 C2_Demand_11: t_11_382 <= 50
 C2_Demand_12: t_12_382 + t_12_13 + t_12_11 <= 180
 C2_Demand_13: t_13_382 <= 30
 C2_Demand_14: t_14_382 <= 75
 C2_Demand_15: t_15_382 <= 80
 C2_Demand_382: t_382_382 <= 0
Bounds
Generals
 t_1_382 t_2_382 t_3_382 t_4_382 t_5_382 t_6_382 t_7_382 t_8_382 t_9_382
 t_10_382 t_11_382 t_12_382 t_13_382 t_14_382 t_15_382 t_382_382 t_3_4
 t_3_5 t_6_7 t_7_8 t_6_8 t_9_10 t_12_13 t_12_11
End
