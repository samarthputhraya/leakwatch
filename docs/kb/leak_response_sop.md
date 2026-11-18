---
short: LeakWatch SOP
title: LeakWatch leak response standard operating procedure
---

# LeakWatch leak response standard operating procedure

This is the procedure the LeakWatch demo follows from an alert to a closed repair. It was written
by the project team for the L-Town benchmark network (a fictional town used in the BattLeDIM
competition). It is not an official utility procedure; for the official guidance on leak
detection and water audit see the CPHEEO Manual on Operation and Maintenance of Water Supply
Systems (2005), Chapter 15 "Water Audit and Leakage Control".

## 1. Scope and roles

- The control-room operator watches the alert feed, triages each alert, sets its priority and
  sends the work order.
- The crew lead receives the work order, inspects the listed pipes in order, confirms the leak and
  organises the repair.
- The utility engineer reviews false alarms and recurring alerts and tunes the detector.
- Residents report problems; their complaints are matched to alerts (in the demo, complaints are
  synthetic because L-Town is fictional).

## 2. Triage a new alert

When a new alert appears, check in this order:

1. Status and timing: when the alert fired (start time) and whether it is still open.
2. Strength: the K-Means score against the threshold. An alert fires only after several
   consecutive one-hour windows score above the threshold, so a single noisy reading does not
   raise one.
3. Where: the area (A, B or C) and the top sensors, the sensors that explain most of the anomaly
   score. If localizer candidates exist, look at the five candidate nodes on the map.
4. Size: the estimated loss rate (excess inflow against the usual inflow for that hour of the
   week). It is an estimate, not a measured leak flow.
5. Related events: other open alerts in the same area, resident complaints, planned works, pump
   or valve operations that could change pressures.

## 3. Rule out a sensor fault

Before sending a crew, make sure the alert is not caused by a broken sensor:

- Open the chart of each top sensor around the alert. A real leak usually lowers pressure at
  several nearby sensors at the same time and raises the inflow at the reservoir inlets.
- A sensor that is stuck (the same value for an hour or more), jumps to an impossible value, or
  moves alone while its neighbours stay normal points to a sensor fault, not a leak. The cleaned
  readings table flags stuck and out-of-range values.
- Pump switching (PUMP_1) and tank filling change pressures in area C every day; compare with the
  hour-of-week baseline before calling it a leak.
- If the evidence points to a sensor fault, mark the alert false_alarm, note the sensor and raise
  a maintenance ticket for it.

## 4. Set the priority

LeakWatch sets the priority of a work order from two numbers (the levels are operator policy in
`configs/default.yaml`, section `copilot.dispatch.priority`):

- P1 (send a crew now): estimated loss of at least 10 m3/h, or an anomaly score at least 3 times
  the threshold.
- P2 (send a crew within the shift): estimated loss of at least 3 m3/h, or a score at least 1.5
  times the threshold.
- P3 (plan an inspection): everything else, including small leaks that grow slowly.

Raise the priority by hand when the leak threatens a road, a building or a critical customer
(hospital, school), or when residents report no water.

## 5. Send the work order

Every alert gets a work order with: priority, a short summary, the pipes to inspect in order, the
estimated loss per hour, the evidence it is based on, recommended actions and citations from the
manuals.

- When the localizer (GNN) has produced candidates, the pipes attached to the five most likely
  nodes come first, most probable node first.
- Without candidates, the pipes nearest the top sensors (by distance along the network) are
  listed, starting with the top sensor.
- If the language model is not available, the work order is produced from the same evidence with
  a fixed template and is marked "template fallback". It is still a valid work order.

## 6. Confirm the leak before digging

Never excavate on the alert alone. The crew confirms the leak first:

1. Walk the listed pipes in order and look for signs at the surface: water on the road, wet
   patches in dry weather, sinking ground, unusual plant growth, water in drains or valve chambers.
2. Listen for leak noise at valves, hydrants and meters with a listening stick. A leak on a
   pressurised pipe makes a hissing or rushing sound that is loudest near the leak.
3. Narrow the spot with a ground microphone along the pipe route, or with a leak noise correlator
   between two fittings on either side of the suspected leak.
4. Where the location is still unclear, run a step test at night: close valves one section at a
   time while watching the inflow meter. The section whose closure drops the inflow holds the leak.
5. Mark the spot and check the records for other buried services before digging.

The CPHEEO O&M manual (Chapter 15, sections 15.10 "Leakage detection and monitoring" and 15.11
"Assessment of leakage") describes leak detection and leakage assessment in detail.

## 7. Isolate, repair and close the alert

- If a burst is confirmed, close the nearest valves to isolate the section and tell the control
  room before the repair starts, so that the pressure change is not read as a new leak.
- After the repair, open the valves slowly, flush the section and check that pressures at the
  top sensors return to their baseline.
- Set the alert status to confirmed (leak found and repaired) or false_alarm (nothing found),
  record the repair time and the pipe repaired, and close the work order.
- Confirmed leaks and false alarms are kept: they are used to check and tune the detector.

## 8. Resident complaints

- Record the complaint with the area, the type of problem (no water, low pressure, visible leak,
  noise, water quality) and when it started.
- LeakWatch matches the complaint to alerts in the same area that were active around that time and
  replies with the status: linked to an alert, logged for review, or a request for the area or
  street if it was not given.
- A complaint with no matching alert is still reviewed by an operator: residents often notice a
  leak before the sensors do, especially in area B, which has a single pressure sensor.
- Water quality complaints (dirty or smelly water) can follow a burst or a repair; send them to
  the water quality team as well.

## 9. When the copilot or the model is unavailable

- Alerts, charts and template work orders do not need the language model; only chat answers do.
- If chat answers fail, use the alert view and the work order directly, and look up procedures in
  this document or in the CPHEEO manual.
- Every number the copilot states comes from a tool result. If an answer looks wrong, open the
  tool call details (the SQL that ran) and check the source table.
