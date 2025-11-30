

IMG_PATH="Your Image Save Path"
FLUX_PATH=models/FLUX.1-dev
python batch_generate.py \
    --model_name_or_path models/X-Omni-En \
    --flux_model_name_or_path $FLUX_PATH \
    --prompt_file prompts/Geneval++.jsonl \
    --image-size 576 576 \
    --cfg-scale 1.0 \
    --min-p 0.03 \
    --seed 1234 \
    --output-path $IMG_PATH \
    --lora_weights_path GvU_ckpts/ckpt.pt \