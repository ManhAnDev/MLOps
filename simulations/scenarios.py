from simulator import PredictionSimulator

sim = PredictionSimulator()

scenarios = {
    1: ("normal", 200),
    2: ("moderate_drift", 200),
    3: ("severe_drift", 200),
}

import sys

if len(sys.argv) > 1:
    s = int(sys.argv[1])
    scenario, n = scenarios[s]
    sim.run_simulation(n_requests=n, scenario=scenario)
else:
    for scenario, n in scenarios.values():
        sim.run_simulation(n_requests=n, scenario=scenario)
