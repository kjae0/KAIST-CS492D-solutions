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
  edit_prompt=$(jq -r --arg key "$key" '.[$key].edit_prompt' "$json_file")
  img_path=$(jq -r --arg key "$key" '.[$key].img_path' "$json_file")
  
  # Run the Python script with the extracted prompt
  echo "Running: python main.py --prompt \"$prompt\""
  python main.py --prompt "$prompt" --loss_type sds --guidance_scale 25 --save_dir "./sds_output/$key"
  CUDA_VISIBLE_DEVICES="0" python main.py --prompt "$prompt" --loss_type pds --guidance_scale 7.5 --edit_prompt "$edit_prompt" --src_img_path "./data/imgs/$img_path" --save_dir "./pds_output/$key"
done
