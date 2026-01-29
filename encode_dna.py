#!/usr/bin/env python
"""
Standalone DNA encoding script using rate 1/2 convolutional code.

CLI Usage:
    python encode_dna.py --input raw_dna.txt --output encoded_dna.txt

Module Usage:
    from encode_dna import encode_sequence, encode_sequences

    encoded = encode_sequence("ACGTACGT...")  # Single sequence
    encoded_list = encode_sequences(["ACGT...", "GCTA..."])  # Multiple
"""

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend before importing pyplot

import numpy as np
import argparse
from conv_code import conv_code

# Generator matrix for rate 1/2 quaternary convolutional code (memory 2)
DEFAULT_G = np.array([[1, 1, 1], [1, 2, 1]])


def dna_to_quat(dna_str):
    """Convert DNA string to quaternary array (A=0, C=1, G=2, T=3)."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    return np.array([mapping[base] for base in dna_str.upper()])


def quat_to_dna(quat_array):
    """Convert quaternary array to DNA string."""
    return ''.join(['ACGT'[x] for x in quat_array])


def _build_encoder(in_len, G=None):
    """Build convolutional code encoder for given input length."""
    if G is None:
        G = DEFAULT_G

    cc = conv_code()
    cc.quar_cc(G)
    cc.make_trellis(in_len)
    cc.make_encoder()
    return cc


def encode_sequence(dna_seq, G=None):
    """
    Encode a single DNA sequence using rate 1/2 convolutional code.

    Args:
        dna_seq: DNA string (e.g., "ACGTACGT")
        G: Generator matrix (default: rate 1/2, memory 2)

    Returns:
        Encoded DNA string (~2x length of input)

    Note:
        Input length should be EVEN for later Trellis BMA reconstruction.
    """
    if len(dna_seq) % 2 != 0:
        print(f"Warning: Input length {len(dna_seq)} is ODD. Trellis BMA requires EVEN length.")

    in_quat = dna_to_quat(dna_seq)
    cc = _build_encoder(len(in_quat), G)
    out_quat = cc.encode(in_quat)
    return quat_to_dna(out_quat)


def encode_sequences(dna_sequences, G=None):
    """
    Encode multiple DNA sequences.

    Args:
        dna_sequences: List of DNA strings
        G: Generator matrix (default: rate 1/2, memory 2)

    Returns:
        List of encoded DNA strings
    """
    if not dna_sequences:
        return []

    # Group sequences by length for encoder reuse
    by_length = {}
    for i, seq in enumerate(dna_sequences):
        length = len(seq)
        if length not in by_length:
            by_length[length] = []
        by_length[length].append((i, seq))

    # Encode each group
    results = [None] * len(dna_sequences)

    for length, group in by_length.items():
        if length % 2 != 0:
            print(f"Warning: Found sequences with ODD length {length}. Trellis BMA requires EVEN length.")

        cc = _build_encoder(length, G)

        for idx, seq in group:
            in_quat = dna_to_quat(seq)
            out_quat = cc.encode(in_quat)
            results[idx] = quat_to_dna(out_quat)

    return results


def encode_file(input_path, G=None):
    """
    Read DNA sequences from file and encode them.

    Args:
        input_path: Path to file with one DNA sequence per line
        G: Generator matrix (default: rate 1/2, memory 2)

    Returns:
        List of encoded DNA strings
    """
    sequences = []
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                sequences.append(line)

    return encode_sequences(sequences, G)


def main():
    parser = argparse.ArgumentParser(
        description='Encode DNA sequences using rate 1/2 convolutional code'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input file with DNA sequences (one per line)'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output file for encoded sequences'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print progress information'
    )

    args = parser.parse_args()

    if args.verbose:
        print(f"Reading sequences from {args.input}...")

    encoded = encode_file(args.input)

    if args.verbose:
        print(f"Encoded {len(encoded)} sequences")
        if encoded:
            print(f"Input length: {len(open(args.input).readline().strip())}")
            print(f"Output length: {len(encoded[0])}")

    with open(args.output, 'w') as f:
        for seq in encoded:
            f.write(seq + '\n')

    if args.verbose:
        print(f"Written to {args.output}")


if __name__ == '__main__':
    main()
