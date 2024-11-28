from src.importation import argparse
from src.experiment import train

parser = argparse.ArgumentParser()
parser.add_argument("--json_data", type=str)
parser.add_argument("--seed", type=float)
parser.add_argument("--pretrain", type=str, default="")
parser.add_argument("--storage_path", type=str, default=None)
args = parser.parse_args()

train(args.json_data, args.storage_path, int(args.seed), args.pretrain)