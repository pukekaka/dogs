### Going through the segments

segments = dict()

# For each segment
for seg_ea in Segments():

    data = []
    
    # For each byte in the address range of the segment
    for ea in range(seg_ea, SegEnd(seg_ea)):

        # Fetch byte
        data.append(chr(Byte(ea)))
    
    # Put the data together
    segments[SegName(seg_ea)] = ''.join(data)

# Loop through the dictionary and print the segment's names
# and their sizes
for seg_name, seg_data in segments.items():
    print seg_name, len(seg_data)
