#!/bin/bash

# Define the local directory where you want to save the PDF files
local_directory="D:/Test/CSPapers/"

# Path to your text file containing the IDs
id_file="cs_LG_ids_1000_new.txt"

# Loop through each line in the file
while IFS= read -r id; do
    # Construct the remote path for the PDF
    remote_path="gs://arxiv-dataset/arxiv/cs/pdf/${id:0:4}/${id}.pdf"

    # Construct the local path for the PDF
    local_path="${local_directory}${id}.pdf"

    # Download the file
    gsutil cp "${remote_path}" "${local_path}"

    # Echo the status
    echo "Downloaded ${id}.pdf to ${local_path}"
done < "$id_file"