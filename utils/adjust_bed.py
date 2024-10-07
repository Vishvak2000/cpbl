import sys

def extend_bed(input_file, output_file, extension=500):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            if line.startswith('#') or not line.strip():
                continue  # Skip headers and empty lines
            
            # Split the line by tabs
            fields = line.strip().split('\t')
            
            # Extract the chromosome, start, and end positions
            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[2])
            
            # Calculate the summit (midpoint)
            summit = (start + end) // 2
            
            # Calculate the new extended coordinates
            new_start = max(0, summit - extension)  # Ensure start is not negative
            new_end = summit + extension
            
            # Prepare the new fields for the output
            new_fields = [chrom, str(new_start), str(new_end)] + fields[3:]
            
            # Write the new line to the output file
            outfile.write('\t'.join(new_fields) + '\n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extend_bed.py <input.bed> <output.bed>")
        sys.exit(1)

    input_bed = sys.argv[1]
    output_bed = sys.argv[2]
    
    extend_bed(input_bed, output_bed)
