#!/usr/bin/env python
"""
Standalone DNA reconstruction script using Trellis BMA / BCJR.

python reconstruct_dna.py --traces recon_input/aging/reads.txt --output recon_output/aging/output-results-combined.txt --train-refs ./recon_input/aging/reference.txt --train-traces ./recon_input/aging/reads.txt --in-len 116 --verbose

CLI Usage:
    # Learn IDS rates from training data:
    python reconstruct_dna.py --traces test.txt --output out.txt \\
        --train-refs train_refs.txt --train-traces train_traces.txt

    # Or specify rates manually:
    python reconstruct_dna.py --traces test.txt --output out.txt \\
        --p-del 0.015 --p-ins 0.012 --p-sub 0.018

    # Choose reconstruction method:
    python reconstruct_dna.py --traces test.txt --output out.txt \\
        --train-refs refs.txt --train-traces traces.txt --method bcjr

Module Usage:
    from reconstruct_dna import estimate_ids_rates, reconstruct_cluster

    # Learn rates from training data
    p_del, p_ins, p_sub = estimate_ids_rates('train_refs.txt', 'train_traces.txt')

    # Reconstruct
    traces = ["ACGT...", "ACGTT...", "CGT..."]
    reconstructed = reconstruct_cluster(traces, in_len=100,
                                        p_del=p_del, p_ins=p_ins, p_sub=p_sub)

Methods:
    - trellis-bma (default): Linear complexity in number of traces, approximate
    - bcjr: Exact algorithm, exponential complexity in number of traces

Note: The input length (--in-len) is the ORIGINAL input length before encoding.
      For rate 1/2 code, encoded length = 2 * in_len.
      The reconstruction returns the ENCODED sequence, not the original input.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend before importing pyplot

import numpy as np
import time
import argparse
import os
from conv_code import conv_code
from coded_ids_multiD import coded_ids_multiD
from trellis_bma import trellis_bma
from Levenshtein import editops

# Generator matrix for rate 1/2 quaternary convolutional code (memory 2)
# Must match the encoding parameters
DEFAULT_G = np.array([[1, 1, 1], [1, 2, 1]])

DEFAULT_MAX_DRIFT = 3
DEFAULT_MAX_TRACES = 7
LOOKAHEAD = False
EXTEND_FORWARD = True

# Global trellis cache: key = (in_len, num_traces, max_drift), value = (cc, ids_trellis)
_TRELLIS_CACHE = {}

def dna_to_quat(dna_str):
    """Convert DNA string to quaternary array (A=0, C=1, G=2, T=3)."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    return np.array([mapping[base] for base in dna_str.upper()])


def quat_to_dna(quat_array):
    """Convert quaternary array to DNA string."""
    return ''.join(['ACGT'[x] for x in quat_array])


def _build_trellis(in_len, num_traces, G=None, p_del=0.02, p_ins=0.02, p_sub=0.02, max_drift=None, use_cache=True):
    """Build the IDS trellis for reconstruction.

    Args:
        use_cache: If True, cache trellis by parameters to avoid rebuilding for identical configs.
    """
    if G is None:
        G = DEFAULT_G
    if max_drift is None:
        max_drift = DEFAULT_MAX_DRIFT

    # Check cache - include error rates since they affect edge weights
    cache_key = (in_len, num_traces, max_drift, p_del, p_ins, p_sub)
    if use_cache and cache_key in _TRELLIS_CACHE:
        return _TRELLIS_CACHE[cache_key]

    # Build convolutional code trellis
    cc = conv_code()
    cc.quar_cc(G)
    cc.make_trellis(in_len)
    cc.make_encoder()

    # Build IDS trellis
    ids_trellis = coded_ids_multiD(
        A_in=4, A_cw=4,
        code_trellis_states=cc.trellis_states,
        code_trellis_edges=cc.trellis_edges,
        code_time_type=cc.time_type,
        num_traces=num_traces,
        p_del=p_del, p_sub=p_sub, p_ins=p_ins,
        max_drift=max_drift
    )

    # Cache the result
    if use_cache:
        _TRELLIS_CACHE[cache_key] = (cc, ids_trellis)

    return cc, ids_trellis


def reconstruct_cluster(traces, in_len=None, G=None, p_del=None, p_ins=None, p_sub=None,
                        max_drift=None, method='trellis-bma', return_probs=False, max_traces=10,
                        extend_forward=False):
    """
    Reconstruct DNA from a cluster of noisy traces.

    Args:
        traces: List of DNA trace strings (e.g., ["ACGT...", "ACGTT...", "CGT..."])
        in_len: Original input length before encoding (MUST be EVEN).
                If None, inferred from trace lengths.
        G: Generator matrix (default: rate 1/2, memory 2)
        p_del: Deletion probability (default: 0.02)
        p_ins: Insertion probability (default: 0.02)
        p_sub: Substitution probability (default: 0.02)
        max_drift: Maximum drift from expected position (default: 10)
        method: 'trellis-bma' (default, linear complexity) or 'bcjr' (exact, exponential)
        return_probs: If True, also return posterior probabilities
        max_traces: Maximum number of traces to use (randomly sampled if exceeded)

    Returns:
        If return_probs=False: Reconstructed DNA string
        If return_probs=True: (reconstructed_dna, posterior_probs)

    Note:
        Returns the reconstructed ENCODED sequence (length ~2 * in_len for rate 1/2).
    """
    if not traces:
        raise ValueError("No traces provided")

    # Convert all traces to quaternary
    import random
    trace_list = [dna_to_quat(t) for t in traces]

    # Infer input length from traces if not provided
    if in_len is None:
        # For rate 1/2 code: encoded_len = 2 * in_len
        # Traces should be close to encoded_len
        avg_trace_len = np.mean([len(t) for t in trace_list])
        in_len = int(round(avg_trace_len / 2))
        if in_len % 2 != 0:
            in_len += 1  # Ensure even

    if in_len % 2 != 0:
        raise ValueError(f"Input length {in_len} must be EVEN for Trellis BMA")

    # Expected codeword length (for rate 1/2)
    N_cw = 2 * in_len

    if max_drift is None:
        max_drift = DEFAULT_MAX_DRIFT

    # Prioritize traces within max_drift, then fill with force-normalized traces if needed
    good_traces = [tr for tr in trace_list if np.abs(len(tr) - N_cw) <= max_drift]
    bad_traces = [tr for tr in trace_list if np.abs(len(tr) - N_cw) > max_drift]

    if len(good_traces) >= max_traces:
        selected = random.sample(good_traces, max_traces)
    else:
        # Use all good traces, fill remaining slots with force-normalized bad traces
        selected = list(good_traces)
        remaining = max_traces - len(selected)
        if remaining > 0 and bad_traces:
            fill = random.sample(bad_traces, min(remaining, len(bad_traces)))
            selected.extend(fill)

    # Normalize any traces outside max_drift
    normalized_traces = []
    for tr in selected:
        if np.abs(len(tr) - N_cw) <= max_drift:
            normalized_traces.append(tr)
        elif len(tr) > N_cw:
            idx_delete = np.random.choice(len(tr), len(tr) - N_cw - max_drift, replace=False)
            normalized_traces.append(np.delete(tr, idx_delete))
        else:
            idx_insert = np.random.choice(len(tr), N_cw - max_drift - len(tr), replace=False)
            normalized_traces.append(np.insert(tr, idx_insert, 0))

    # IMPORTANT: Trellis BMA processes traces one at a time, so always use num_traces=1
    cc, ids_trellis = _build_trellis(
        in_len, num_traces=1, G=G, p_del=p_del, p_ins=p_ins, p_sub=p_sub, max_drift=max_drift
    )

    # Run reconstruction algorithm
    if method == 'bcjr':
        post_probs = ids_trellis.bcjr(
            normalized_traces,
            cc.trellis_states[0][0],
            cc.trellis_states[-1]
        )
        estimate = post_probs.argmax(axis=1)
    elif method == 'trellis-bma':
        estimate, post_probs = trellis_bma(
            ids_trellis,
            normalized_traces,
            cc.trellis_states[0][0],
            cc.trellis_states[-1],
            lookahead=LOOKAHEAD,
            extend_forward=extend_forward
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'bcjr' or 'trellis-bma'")

    reconstructed_dna = quat_to_dna(estimate)

    if return_probs:
        return reconstructed_dna, post_probs
    return reconstructed_dna


def reconstruct_clusters(clusters, in_len=None, G=None, p_del=None, p_ins=None, p_sub=None,
                         max_drift=None, method='trellis-bma', verbose=False, max_traces=10,
                         extend_forward=False):
    """
    Reconstruct DNA from multiple clusters.

    Args:
        clusters: List of clusters, where each cluster is a list of DNA trace strings
        in_len: Original input length before encoding (MUST be EVEN)
        G: Generator matrix (default: rate 1/2, memory 2)
        p_del: Deletion probability (default: 0.02)
        p_ins: Insertion probability (default: 0.02)
        p_sub: Substitution probability (default: 0.02)
        max_drift: Maximum drift from expected position (default: 10)
        method: 'trellis-bma' (default, linear complexity) or 'bcjr' (exact, exponential)
        verbose: Print progress
        max_traces: Maximum number of traces to use per cluster

    Returns:
        List of reconstructed DNA strings
    """
    results = []
    first_cluster = True

    for i, cluster in enumerate(clusters):
        if verbose and len(cluster) > max_traces:
            print(f"  Cluster {i+1} has {len(cluster)} traces, subsampling to {max_traces}")

        cluster_start = time.time()
        reconstructed = reconstruct_cluster(
            cluster, in_len, G, p_del, p_ins, p_sub, max_drift, method, max_traces=max_traces,
                extend_forward=extend_forward
        )
        results.append(reconstructed)

        if verbose:
            cache_status = " (built trellis + JIT)" if first_cluster else " (cached)"
            print(f"  Processed {i + 1}/{len(clusters)} clusters in {time.time() - cluster_start:.2f}s{cache_status}")
            first_cluster = False

    return results


def parse_trace_file(filepath):
    """
    Parse trace file with clusters separated by ===== lines.

    Returns:
        List of clusters, where each cluster is a list of DNA strings
    """
    clusters = []
    current_cluster = []

    with open(filepath, 'r') as f:
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

    return clusters


def estimate_ids_rates(reference_file, traces_file, max_training_clusters=None):
    """
    Estimate IDS channel parameters (p_del, p_ins, p_sub) from training data.

    Compares reference sequences against their noisy traces using Levenshtein
    edit operations to count deletions, insertions, and substitutions.

    Args:
        reference_file: Path to file with reference sequences (one per line)
        traces_file: Path to file with clustered traces (===== separated)
        max_training_clusters: Max clusters to use for training (None = all)

    Returns:
        (p_del, p_ins, p_sub) tuple of estimated probabilities
    """
    # Load references
    references = []
    with open(reference_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                references.append(line)

    # Load trace clusters
    clusters = parse_trace_file(traces_file)

    training_size = len(clusters)
    if max_training_clusters is not None:
        training_size = min(training_size, max_training_clusters)

    if training_size == 0:
        raise ValueError("No training clusters found")
    if len(references) < training_size:
        raise ValueError(f"Not enough references ({len(references)}) for {training_size} clusters")

    # Count edit operations
    num_del = 0
    num_ins = 0
    num_sub = 0
    num_traces = 0

    for idx in range(training_size):
        ref = references[idx]
        for trace in clusters[idx]:
            num_traces += 1
            ops = editops(ref, trace)
            if len(ops) > 0:
                ops_array = np.array(ops)
                num_del += (ops_array[:, 0] == 'delete').sum()
                num_ins += (ops_array[:, 0] == 'insert').sum()
                num_sub += (ops_array[:, 0] == 'replace').sum()

    if num_traces == 0:
        raise ValueError("No traces found in training data")

    # Normalize by total symbol positions
    total_symbols = num_traces * len(references[0])
    p_del = num_del / total_symbols
    p_ins = num_ins / total_symbols
    p_sub = num_sub / total_symbols

    return p_del, p_ins, p_sub


def main():
    parser = argparse.ArgumentParser(
        description='Reconstruct DNA from noisy traces using Trellis BMA / BCJR'
    )
    parser.add_argument(
        '--traces', '-t',
        required=True,
        help='Input file with clustered traces (separated by ===== lines)'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output file for reconstructed sequences'
    )
    parser.add_argument(
        '--in-len', '-n',
        type=int,
        default=None,
        help='Original input length before encoding (must be EVEN). If not specified, inferred from traces.'
    )
    parser.add_argument(
        '--train-refs',
        help='Training reference sequences file (one per line). Use with --train-traces to learn IDS rates.'
    )
    parser.add_argument(
        '--train-traces',
        help='Training traces file (===== separated clusters). Use with --train-refs to learn IDS rates.'
    )
    parser.add_argument(
        '--p-del',
        type=float,
        default=None,
        help='Deletion probability (manual override, alternative to --train-refs/--train-traces)'
    )
    parser.add_argument(
        '--p-ins',
        type=float,
        default=None,
        help='Insertion probability (manual override, alternative to --train-refs/--train-traces)'
    )
    parser.add_argument(
        '--p-sub',
        type=float,
        default=None,
        help='Substitution probability (manual override, alternative to --train-refs/--train-traces)'
    )
    parser.add_argument(
        '--max-drift',
        type=int,
        default=DEFAULT_MAX_DRIFT,
        help=f'Maximum drift from expected position (default: {DEFAULT_MAX_DRIFT})'
    )
    parser.add_argument(
        '--max-traces',
        type=int,
        default=DEFAULT_MAX_TRACES,
        help='Maximum number of traces to use per cluster (default: 10). Larger clusters will be randomly subsampled.'
    )
    parser.add_argument(
        '--method', '-m',
        choices=['trellis-bma', 'bcjr'],
        default='trellis-bma',
        help='Reconstruction method: trellis-bma (default, linear complexity) or bcjr (exact, exponential)'
    )
    parser.add_argument(
        '--extend-forward',
        action='store_true',
        default=EXTEND_FORWARD,
        help='Extend forward propagation past midpoint and use f*b for backward decisions. '
             'Fixes last-position errors at ~1.5x cost (vs 3x for full lookahead). Default: off.'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print progress information'
    )

    args = parser.parse_args()

    if args.in_len is not None and args.in_len % 2 != 0:
        parser.error("--in-len must be EVEN for Trellis BMA reconstruction")

    # Determine IDS rates
    has_training = args.train_refs and args.train_traces
    has_manual = args.p_del is not None or args.p_ins is not None or args.p_sub is not None

    if has_training and has_manual:
        parser.error("Provide either --train-refs/--train-traces OR manual rates (--p-del, etc.), not both")
    elif has_training:
        if not args.train_refs or not args.train_traces:
            parser.error("Both --train-refs and --train-traces are required for rate learning")
        if args.verbose:
            print(f"Learning IDS rates from training data...")
        start = time.time()
        p_del, p_ins, p_sub = estimate_ids_rates(args.train_refs, args.train_traces)
        if args.verbose:
            print(f"Time taken for estimating rates: {time.time() - start:.2f} seconds")
        if args.verbose:
            print(f"Learned rates: p_del={p_del:.4f}, p_ins={p_ins:.4f}, p_sub={p_sub:.4f}")
    elif has_manual:
        p_del = args.p_del if args.p_del is not None else 0.0
        p_ins = args.p_ins if args.p_ins is not None else 0.0
        p_sub = args.p_sub if args.p_sub is not None else 0.0
    else:
        parser.error("Must provide either --train-refs/--train-traces OR manual rates (--p-del, --p-ins, --p-sub)")

    if args.verbose:
        print(f"Reading traces from {args.traces}...")

    clusters = parse_trace_file(args.traces)

    if args.verbose:
        print(f"Found {len(clusters)} clusters")
        print(f"Cluster sizes: min={min(len(c) for c in clusters)}, max={max(len(c) for c in clusters)}")
        print(f"Parameters: p_del={p_del:.4f}, p_ins={p_ins:.4f}, p_sub={p_sub:.4f}, max_drift={args.max_drift}, max_traces={args.max_traces}")
        print(f"Method: {args.method}")
        print("Reconstructing...")
        print("  Note: First cluster will be slow (trellis construction + JIT compilation)")
        print("        Subsequent clusters will reuse cached trellis (built for 1 trace) and compiled code")

    start = time.time()
    reconstructed = reconstruct_clusters(
        clusters,
        in_len=args.in_len,
        p_del=p_del,
        p_ins=p_ins,
        p_sub=p_sub,
        max_drift=args.max_drift,
        method=args.method,
        verbose=args.verbose,
        max_traces=args.max_traces,
        extend_forward=args.extend_forward
    )
    print(f"Time taken for reconstruction: {time.time() - start:.2f} seconds")

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output, 'w') as f:
        for seq in reconstructed:
            f.write(seq + '\n')

    if args.verbose:
        print(f"Written {len(reconstructed)} sequences to {args.output}")


if __name__ == '__main__':
    main()
