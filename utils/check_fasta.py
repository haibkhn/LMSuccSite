import re

def count_specific_sequences(fasta_file_path, id_list_path):
    """
    Count how many times each ID from the list appears in the FASTA file
    """
    # Read IDs to check
    with open(id_list_path, 'r') as f:
        ids_to_check = [line.strip() for line in f]
    
    # Initialize counters for each ID
    id_counts = {id_: 0 for id_ in ids_to_check}
    
    # Process FASTA file
    with open(fasta_file_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                # Extract ID between first and second pipe
                match = re.search(r'\|(.*?)\|', line)
                if match:
                    uniprot_id = match.group(1)
                    # If this ID is in our list, increment its counter
                    if uniprot_id in id_counts:
                        id_counts[uniprot_id] += 1
    
    # Print results
    print("\nSequence Counts:")
    print("-" * 30)
    total_sequences = 0
    for id_ in ids_to_check:
        count = id_counts[id_]
        total_sequences += count
        print(f"{id_}: {count} sequences")
    
    print(f"\nTotal sequences for listed IDs: {total_sequences}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python script.py <fasta_file> <id_list_file>")
        sys.exit(1)
    
    try:
        count_specific_sequences(sys.argv[1], sys.argv[2])
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
    except Exception as e:
        print(f"Error: {e}")