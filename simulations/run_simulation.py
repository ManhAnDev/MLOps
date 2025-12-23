import argparse
from simulator import PredictionSimulator

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--requests", type=int, default=100)
parser.add_argument("-r", "--rps", type=float, default=2.0)
parser.add_argument("-s", "--scenario", default="normal")

args = parser.parse_args()

sim = PredictionSimulator()
sim.run_simulation(
    n_requests=args.requests,
    scenario=args.scenario,
    rps=args.rps,
)
