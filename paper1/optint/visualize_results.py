# We created this `visualize_results.py` for visualization the result

import pickle
from visualize import draw, draw_spectrum  # Import functions from visualize.py
import matplotlib.pyplot as plt
import numpy as np

# Load the data from the .pkl files
def load_pickle_file(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# Define paths to your files (replace with actual paths)
problems_path = 'results/path-std_nnodes5_noise1.00_S2_target[3, 4]/problems.pkl'
As_path = 'results/path-std_nnodes5_noise1.00_S2_target[3, 4]/As.pkl'
Probs_path = 'results/path-std_nnodes5_noise1.00_S2_target[3, 4]/Probs.pkl'

# Load data
problems = load_pickle_file(problems_path)
As = load_pickle_file(As_path)
Probs = load_pickle_file(Probs_path)

# Inspect a single problem instance
problem = problems[0]  # Visualize the first problem instance

# Define sets based on the problem attributes (assuming problem has these attributes)
colored_set = set(np.where(problem.a_target != 0)[0])  # Nodes with interventions
solved_set = set()  # Any solved nodes, if applicable
affected_set = set().union(*[problem.DAG.descendants_of(i) for i in colored_set])  # Descendants of target nodes

# Draw the DAG structure
draw(
    pdag=problem.DAG,
    colored_set=colored_set,
    solved_set=solved_set,
    affected_set=affected_set
)

# Optional: Draw the spectrum if A and B matrices are available
try:
    draw_spectrum(problem.A, problem.B)
except AttributeError:
    print("Attributes A and B not found in the problem instance.")
