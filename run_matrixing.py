from src.importation import argparse
from src.infer_matrix import infer_matrix

parser = argparse.ArgumentParser()
parser.add_argument("--training_run", type=str)
parser.add_argument("--dataset", type=str)
parser.add_argument("--seed", type=float)
parser.add_argument("--storage_path", type=str, default=None)
args = parser.parse_args()

infer_matrix(
    args.training_run,
    "mogi_standard",
    int(args.seed),
    128,
    args.storage_path,
)