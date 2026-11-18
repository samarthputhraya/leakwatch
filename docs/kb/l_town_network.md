---
short: L-Town sensors and areas
title: L-Town network, sensors and areas
---

# L-Town network, sensors and areas

L-Town is the benchmark network of BattLeDIM 2020 (Battle of the Leakage Detection and Isolation
Methods). It is a fictional town built from a real network layout. All counts below were measured
from the network model file `L-TOWN.inp` by `leakwatch.data.network` and are stored in
`reports/metrics/network_stats.json`.

## 1. The network

- 782 junctions, 2 reservoirs (R1 and R2), 1 tank (T1), 905 pipes, 1 pump (PUMP_1) and 3
  pressure-reducing valves (PRV-1, PRV-2, PRV-3).
- Total pipe length 43.163 km; the mean pipe is 47.69 m long.
- Water enters from the two reservoirs through pipes p227 and p235. PUMP_1 lifts water from node
  n54 into tank T1.
- The SCADA data has a reading every 5 minutes for the full years 2018 and 2019.

## 2. Areas

The network splits into three areas at the pump and the pressure-reducing valves. LeakWatch names
them A, B and C, and every node, pipe and sensor belongs to exactly one of them. An alert's area is
the area of its top sensor.

| Area | Junctions | Pipes | Pipe length | Pressure sensors | Other |
|---|---|---|---|---|---|
| A | 659 | 764 | 36.34 km | 29 | reservoirs R1, R2; inflow meters p227, p235 |
| B | 31 | 32 | 1.426 km | 1 (n215) | fed through PRV-3 |
| C | 92 | 109 | 5.398 km | 3 (n1, n4, n31) | tank T1, 82 AMR meters, fed by PUMP_1 |

## 3. Area A

Area A is the main part of the town and holds most of the network. Its 29 pressure sensors are
n54, n105, n114, n163, n188, n229, n288, n296, n332, n342, n410, n415, n429, n458, n469, n495,
n506, n516, n519, n549, n613, n636, n644, n679, n722, n726, n740, n752 and n769. The two reservoir
inlets (p227, p235) measure all water entering the town, which is why the estimated loss of an
alert is based on them.

## 4. Area B

Area B is a small pressure zone behind valve PRV-3 with a single pressure sensor, n215. Because
the valve holds the pressure there steady, a leak in area B changes the reading at n215 only a
little, and with one sensor the detector cannot say where in the area the leak is. Resident
complaints are especially useful for area B.

## 5. Area C

Area C is the zone fed by PUMP_1 through tank T1. It has three pressure sensors (n1, n4 and n31),
the tank level sensor T1 and all 82 automated meter reading (AMR) meters. Pressures in area C
follow the pump and the tank level during the day, so compare readings with the hour-of-week
baseline before calling a change a leak.

## 6. Sensor types and units

| Type | Count | Unit | Ids in the data |
|---|---|---|---|
| pressure | 33 | m (metres of water) | node ids such as n54, n215, n31 |
| flow | 3 | m3/h | p227, p235 (reservoir inlets) and PUMP_1 |
| level | 1 | m | T1 (tank) |
| amr | 82 | L/h | node ids of the meters in area C |

- In the readings tables a sensor is identified by sensor_id plus sensor_type: a few nodes (n1, n4,
  n31) carry both a pressure sensor and an AMR meter.
- In feature tables and in an alert's top sensors, a series id joins the two with a double
  underscore: pressure__n54, flow__p227, level__T1. The summed AMR consumption of area C is the
  series amr_sum__C.

## 7. What the data does not contain

- No customer names, streets or addresses: L-Town is fictional, so resident complaints in the demo
  are synthetic and refer to areas or nodes.
- No valve or pump operation log; pump switching is visible only through the PUMP_1 flow and the
  tank level.
- The leak list of the benchmark (ground truth) is used to evaluate the detector, never to raise
  alerts.
