import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend before importing pyplot

from conv_code import *
from coded_ids_multiD import *
from trellis_bma import *
import numpy as np
import os
from Levenshtein import editops
import time

# Configuration
INPUT_FILE = 'input/input_450_converted.txt'
REFERENCE_FILE = 'input/input_450_reference.txt'
OUTPUT_FILE = 'output/450_reconstructed_sequences.txt'
in_len = -1  # Expected sequence length (must be EVEN)
N_cw = 110    # Codeword length (same as input for uncoded)
max_drift = 5  # Reduced from 15 - your data has low error rates (~1.3%)
MAX_TRACES_TO_USE = 5  # Cap traces per cluster (diminishing returns beyond ~10)

# 1. Read reference sequences (ground truth)
print("Reading reference sequences...")
centers_str = []
with open(REFERENCE_FILE) as f:
    for line in f:
        line = line.strip()
        in_len = len(line)
        if line:
            centers_str.append(line)

print(f"Found {len(centers_str)} reference sequences")

# 2. Read clustered DNA sequences
print("Reading input traces...")
clusters = []
current_cluster = []

with open(INPUT_FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith('='):
            if current_cluster:
                clusters.append(current_cluster)
                current_cluster = []
        else:
            current_cluster.append(line)

    # Add last cluster
    if current_cluster:
        clusters.append(current_cluster)

print(f"Found {len(clusters)} trace clusters")

# 3. Convert DNA to quaternary (A=0, C=1, G=2, T=3)
def dna_to_quat(dna_str):
    mapping = {'A':0, 'C':1, 'G':2, 'T':3}
    return np.array([mapping[base] for base in dna_str])

def quat_to_dna(quat_array):
    return ''.join(['ACGT'[x] for x in quat_array])

# Convert reference sequences to quaternary
centers_list = [dna_to_quat(seq) for seq in centers_str]

# 4. Learn IDS parameters from ground truth (following Experiment1.ipynb)
print("Learning IDS parameters from ground truth...")
num_del = 0
num_ins = 0
num_sub = 0
num_traces = 0

# Use first 100 clusters for training (or all if fewer)
training_size = min(100, len(clusters))

for idx in range(training_size):
    seq = centers_str[idx]
    for trace_str in clusters[idx]:
        num_traces += 1
        ops = editops(seq, trace_str)
        if len(ops) > 0:
            ops_array = np.array(ops)
            num_del += (ops_array[:, 0] == 'delete').sum()
            num_ins += (ops_array[:, 0] == 'insert').sum()
            num_sub += (ops_array[:, 0] == 'replace').sum()

p_del = num_del / (num_traces * len(centers_list[0]))
p_ins = num_ins / (num_traces * len(centers_list[0]))
p_sub = num_sub / (num_traces * len(centers_list[0]))
p_cor = 1.0 - (p_del + p_ins + p_sub)

print(f"Trained on {num_traces} traces from {training_size} clusters")
print(f"Error rates: p_del={p_del:.4f}, p_ins={p_ins:.4f}, p_sub={p_sub:.4f}")
print(f"Total error rate: {p_del + p_ins + p_sub:.4f}")

# 5. Set up UNCODED trellis (identity code)
print("Setting up uncoded trellis...")
cc = conv_code()
G = np.array([[1]])  # Identity: output = input (no coding)
cc.quar_cc(G)
cc.make_trellis(in_len)
cc.make_encoder()

# 6. Pre-build IDS trellises for different cluster sizes (to avoid rebuilding each time)
print("Building IDS trellises for different cluster sizes...")
# Determine unique cluster sizes (capped at MAX_TRACES_TO_USE)
cluster_sizes = sorted(set(min(len(cluster), MAX_TRACES_TO_USE) for cluster in clusters))
print(f"Cluster sizes found (capped at {MAX_TRACES_TO_USE}): {cluster_sizes}")

start = time.time()
ids_trellises = {}
for num_traces in cluster_sizes:
    # Aggressively reduce max_drift - your error rate is only 1.3%!
    if num_traces == 1:
        adjusted_max_drift = max_drift  # 5
    elif num_traces == 2:
        adjusted_max_drift = 3
    elif num_traces == 3:
        adjusted_max_drift = 2
    else:
        adjusted_max_drift = 2  # For 4+ traces, very conservative

    ids_trellises[num_traces] = coded_ids_multiD(
        A_in=4, A_cw=4,
        code_trellis_states=cc.trellis_states,
        code_trellis_edges=cc.trellis_edges,
        code_time_type=cc.time_type,
        num_traces=num_traces,
        p_del=p_del, p_sub=p_sub, p_ins=p_ins,
        max_drift=adjusted_max_drift
    )
    print(f"  Built trellis for {num_traces} traces (max_drift={adjusted_max_drift})")
    print(f"  Time taken: {time.time() - start}")

# 7. Process each cluster
print("Processing clusters...")
reconstructed_sequences = []

for idx, cluster in enumerate(clusters):
    # Convert DNA traces to quaternary (use only first MAX_TRACES_TO_USE)
    trace_list = [dna_to_quat(trace) for trace in cluster[:MAX_TRACES_TO_USE]]

    # Get the appropriate trellis for this cluster size
    num_traces = len(trace_list)
    ids_trellis = ids_trellises[num_traces]

    # Get max_drift for this trellis
    adjusted_max_drift = ids_trellis.max_drift

    # Normalize trace lengths (from Experiment1.ipynb)
    traces = []
    for tr in trace_list:
        if np.abs(len(tr) - N_cw) <= adjusted_max_drift:
            traces.append(tr)
        elif len(tr) > N_cw:
            idx_delete = np.random.choice(len(tr), len(tr) - N_cw - adjusted_max_drift, replace=False)
            traces.append(np.delete(tr, idx_delete))
        else:
            idx_insert = np.random.choice(len(tr), N_cw - adjusted_max_drift - len(tr), replace=False)
            traces.append(np.insert(tr, idx_insert, 0))

    # Run BCJR reconstruction
    post_probs = ids_trellis.bcjr(traces, cc.trellis_states[0][0], cc.trellis_states[-1])
    estimate = post_probs.argmax(axis=1)

    # Convert back to DNA
    dna_reconstructed = quat_to_dna(estimate)
    reconstructed_sequences.append(dna_reconstructed)

    if (idx + 1) % 50 == 0:
        print(f"  Processed {idx + 1}/{len(clusters)} clusters")

# 8. Write output
print(f"Writing output to {OUTPUT_FILE}...")
os.makedirs('output', exist_ok=True)
with open(OUTPUT_FILE, 'w') as f:
    for seq in reconstructed_sequences:
        f.write(seq + '\n')

print(f"Done! Reconstructed {len(reconstructed_sequences)} sequences")