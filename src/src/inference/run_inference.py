from src.importation import argparse
from inference.infer import infer

parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("areas")

args = parser.parse_args()

experience = args.name
areas = args.areas

infer(experience, areas)