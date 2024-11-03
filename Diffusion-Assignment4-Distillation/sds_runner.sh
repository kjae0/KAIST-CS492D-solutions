#!/bin/bash

json_file="./data/prompt_img_pairs.json"

if [ ! -f "$json_file" ]; then
  echo "JSON file not found at $json_file"
  exit 1
fi


keys=$(jq -r 'keys[]' "$json_file")

for key in $keys; do
  # Extract the prompt value for the current key
  prompt=$(jq -r --arg key "$key" '.[$key].prompt' "$json_file")
  
  # Run the Python script with the extracted prompt
  echo "Running: python main.py --prompt \"$prompt\""
  CUDA_VISIBLE_DEVICES="1" python main.py --prompt "$prompt" --loss_type sds --guidance_scale 25 --save_dir "./sds_output/$key"
done
