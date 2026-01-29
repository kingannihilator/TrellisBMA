"""
Convert custom format to RobuSeqNet format.

Input format:
- First line: reference sequence
- Line of asterisks: separator
- Multiple noisy reads
- Empty lines separate clusters

Output format:
- reads.txt: Noisy reads separated by "==============================="
- reference.txt: Reference sequences (one per line)
"""

import argparse
import os
from datetime import datetime


def convert_format(input_file, output_dir=None):
    """
    Convert input file format to reads.txt and reference.txt

    Args:
        input_file: Path to input file
        output_dir: Directory for output files (default: creates timestamped folder in examples/data)
    """
    if output_dir is None:
        # Get the script's directory and construct path to examples/data
        script_dir = os.path.dirname(os.path.abspath(__file__))
        examples_data_dir = os.path.join(script_dir, 'examples', 'data')

        # Create timestamped folder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        input_basename = os.path.splitext(os.path.basename(input_file))[0]
        output_dir = os.path.join(examples_data_dir, f"{input_basename}_{timestamp}")

        # Create the directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")

    reads_output = os.path.join(output_dir, 'reads.txt')
    reference_output = os.path.join(output_dir, 'reference.txt')

    with open(input_file, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    references = []
    all_reads = []

    i = 0
    while i < len(lines):
        # Skip empty lines
        if not lines[i]:
            i += 1
            continue

        # Read reference sequence
        reference = lines[i]
        i += 1

        # Skip asterisks line
        if i < len(lines) and lines[i].startswith('*'):
            i += 1

        # Read noisy reads until we hit empty lines or end of file
        cluster_reads = []
        while i < len(lines) and lines[i] and not lines[i].startswith('*'):
            cluster_reads.append(lines[i])
            i += 1

        # Only add cluster if we have reads
        if cluster_reads:
            references.append(reference)
            all_reads.append(cluster_reads)

    # Write reads.txt
    with open(reads_output, 'w') as f:
        for i, reads in enumerate(all_reads):
            for read in reads:
                f.write(read + '\n')
            # Add separator after each cluster (including the last one)
            f.write('===============================\n')

    # Write reference.txt
    with open(reference_output, 'w') as f:
        for ref in references:
            f.write(ref + '\n')

    print(f"Conversion complete!")
    print(f"  Clusters processed: {len(references)}")
    print(f"  Output files:")
    print(f"    - {reads_output}")
    print(f"    - {reference_output}")

    return reads_output, reference_output


def main():
    parser = argparse.ArgumentParser(
        description='Convert DNA sequence file to Trellis BMA format'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Input file path'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: same as input file)'
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        return

    convert_format(args.input_file, args.output_dir)


if __name__ == '__main__':
    main()
