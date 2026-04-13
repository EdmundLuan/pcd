#!/bin/bash

set -euo pipefail


# Script to perform coupled projected diffusion between multiple diffusion models.

# ===== Start of Configuration =============================================================================
# ==========================================================================================================

# misc
SEED=42
DETERMINISTIC=True
DEVICE="cuda:0"   # cuda:0 | cuda | cpu


# LDM
# SD_MODEL_ID="stabilityai/stable-diffusion-2-1-base"  ## NOTE: the original model is no longer available as of April 2026, below are alternative forked versions of it.
# SD_MODEL_ID="Manojb/stable-diffusion-2-1-base"
SD_MODEL_ID="PIEthonista/stable-diffusion-2-1-base"
SD_NUM_STEPS=100 # 1-999
SD_SAMPLE_HEIGHT=512
SD_SAMPLE_WIDTH=512
SD_LATENT_HEIGHT=64
SD_LATENT_WIDTH=64
SD_LATENT_CHANNELS=4
SD_MODEL_MIN=-1.0
SD_MODEL_MAX=1.0
VAE_SAMPLE_POSTERIOR=True
SD_DTYPE="bfloat16"
SD_IS_TEXT_CONDITIONAL=True
# SD_PROMPT="High-resolution passport photo of a person, facing forward with a neutral expression. Wearing a plain t-shirt, with a clean white background and even, soft lighting. The composition is centered and symmetrical, with the head at the center of the frame."
# SD_CFG_GUIDANCE_SCALE=10.0
SD_PROMPT=""
SD_CFG_GUIDANCE_SCALE=0.0


# Latent Classifier for LDM (to provide gradients for coupling)
CLASSIFIER_CONFIG="model_weights/ldm/stable-diffusion-2-1-base/base_latent_classifier_resnet_enc_multihead_timecond_False_2025-07-21_01-27-39/FFHQA_ldm_latent_classifier_resnet_enc_multihead_sd-2-1-base-512.yaml"
CLASSIFIER_WEIGHTS="model_weights/ldm/stable-diffusion-2-1-base/base_latent_classifier_resnet_enc_multihead_timecond_False_2025-07-21_01-27-39/checkpoints_latent_classifier/best_latent_classifier.pt"


# Image Classifier for analysis / evaluation
IMAGE_CLASSIFIER_CONFIG="model_weights/FFHGA_classifier_resnet_enc_multihead_age_group_1024_gender_128/FFHQA_classifier_resnet_enc_multihead.yaml"
IMAGE_CLASSIFIER_WEIGHTS="model_weights/FFHGA_classifier_resnet_enc_multihead_age_group_1024_gender_128/checkpoints_classifier/best_classifier.pt"



# Convex Hull Projection Exemplars
# We use projection to enforce age group, and coupling to enforce gender
EXEMPLARS_A=(
    "data/images/custom512x512_matte_white_background/exemplars_male_2/01_male_mid.png"
    "data/images/custom512x512_matte_white_background/exemplars_male_2/02_male_old.png"
    # "data/images/custom512x512_matte_white_background/exemplars_male_2/01_male_mid_3.png"
    # "data/images/custom512x512_matte_white_background/exemplars_male_2/02_male_old_3.png"
    # "data/images/custom512x512_matte_white_background/exemplars_male_2/01_male_mid_4.png"
    # "data/images/custom512x512_matte_white_background/exemplars_male_2/02_male_old_4.png"
)
EXEMPLARS_B=(
    "data/images/custom512x512_matte_white_background/exemplars_female_2/01_female_mid.png"
    "data/images/custom512x512_matte_white_background/exemplars_female_2/02_female_old.png"
    # "data/images/custom512x512_matte_white_background/exemplars_female_2/01_female_mid_3.png"
    # "data/images/custom512x512_matte_white_background/exemplars_female_2/02_female_old_3.png"
    # "data/images/custom512x512_matte_white_background/exemplars_female_2/01_female_mid_4.png"
    # "data/images/custom512x512_matte_white_background/exemplars_female_2/02_female_old_4.png"
)


# Projection Config
# -----------------
MASTER_ENABLE_PROJECTION=True # set False to disable projection
PROJECTION_SKIP_LAST=False
PROJECTION_VERBOSE=False
PROJECTION_TIMESTEPS="all"
PROJECTION_MD_NUM_ITERS=10000
PROJECTION_MD_LR=0.00001
RESCALE_PROJECTED_LATENT_NORM=False
DIFF_STEP_NOISE_STD_SCALING_FACTOR=20  # auto override to k=1 if MASTER_ENABLE_PROJECTION=False


# Coupling Config
# ---------------
MASTER_ENABLE_COUPLING=True # set False to disable coupling
COUPLING_GRAD_SCALE=450
COUPLING_VAR="age_group"  # gender | age_group
COUPLING_FN_ROT=0  # rotation of classifier logit classes


# Samples / Batch Size
NUM_SAMPLES_TO_GENERATE=25

# Output Dir
MODEL="${SD_MODEL_ID//\//-}"
SETUP_NAME="T${SD_NUM_STEPS}_STD${DIFF_STEP_NOISE_STD_SCALING_FACTOR}"
if [ "$SD_PROMPT" = "" ]; then
    SETUP_NAME="${SETUP_NAME}_noprompt"
else
    SETUP_NAME="${SETUP_NAME}_prompt"
fi
if [ "$MASTER_ENABLE_PROJECTION" = "True" ]; then
    SETUP_NAME="${SETUP_NAME}_PROJ_${PROJECTION_TIMESTEPS}_MD${PROJECTION_MD_NUM_ITERS}-${PROJECTION_MD_LR}_RescaleLatent${RESCALE_PROJECTED_LATENT_NORM}"
fi
if [ "$MASTER_ENABLE_COUPLING" = "True" ]; then
    SETUP_NAME="${SETUP_NAME}_COUPLE_${COUPLING_VAR}_rot${COUPLING_FN_ROT}_grad${COUPLING_GRAD_SCALE}"
fi
OUTPUTS_DIR="outputs/${SETUP_NAME}/${MODEL}"

# ==========================================================================================================
# ===== End of Configuration ===============================================================================










if [ "$DETERMINISTIC" = "True" ]; then
    deterministic="--deterministic"
else
    deterministic=""
fi
if [ "$SD_IS_TEXT_CONDITIONAL" = "True" ]; then
    sd_is_text_conditional="--sd_is_text_conditional"
else
    sd_is_text_conditional=""
fi
if [ "$VAE_SAMPLE_POSTERIOR" = "True" ]; then
    vae_sample_posterior="--vae_sample_posterior"
else
    vae_sample_posterior=""
fi
if [ "$PROJECTION_SKIP_LAST" = "True" ]; then
    projection_skip_last="--projection_skip_last"
else
    projection_skip_last=""
fi
if [ "$PROJECTION_VERBOSE" = "True" ]; then
    projection_verbose="--projection_verbose"
else
    projection_verbose=""
fi
if [ "$RESCALE_PROJECTED_LATENT_NORM" = "True" ]; then
    rescale_projected_latent_norm="--rescale_projected_latent_norm"
else
    rescale_projected_latent_norm=""
fi
if [ "$MASTER_ENABLE_PROJECTION" = "True" ]; then
    enable_projection="--enable_projection"
else
    enable_projection=""
fi
if [ "$MASTER_ENABLE_COUPLING" = "True" ]; then
    enable_coupling="--enable_coupling"
else
    enable_coupling=""
fi
log_dir="$OUTPUTS_DIR/logs"
mkdir -p $log_dir
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="$log_dir/$timestamp.log"



start_time=$(date +"%Y-%m-%d %H:%M:%S")
start_sec=$(date +%s)
echo "Script started at: $start_time" | tee -a "$log_file"


python -m src.main \
    --seed $SEED \
    $deterministic \
    --device "$DEVICE" \
    --sd_model_id "$SD_MODEL_ID" \
    --sd_num_steps $SD_NUM_STEPS \
    --sd_sample_height $SD_SAMPLE_HEIGHT \
    --sd_sample_width $SD_SAMPLE_WIDTH \
    --sd_latent_height $SD_LATENT_HEIGHT \
    --sd_latent_width $SD_LATENT_WIDTH \
    --sd_latent_channels $SD_LATENT_CHANNELS \
    --sd_model_min $SD_MODEL_MIN \
    --sd_model_max $SD_MODEL_MAX \
    --sd_dtype "$SD_DTYPE" \
    $sd_is_text_conditional \
    --sd_prompt "$SD_PROMPT" \
    --sd_cfg_guidance_scale $SD_CFG_GUIDANCE_SCALE \
    $vae_sample_posterior \
    --classifier_config "$CLASSIFIER_CONFIG" \
    --classifier_weights "$CLASSIFIER_WEIGHTS" \
    --exemplars_a "${EXEMPLARS_A[@]}" \
    --exemplars_b "${EXEMPLARS_B[@]}" \
    $projection_skip_last \
    $projection_verbose \
    --projection_timesteps "$PROJECTION_TIMESTEPS" \
    --md_num_iters $PROJECTION_MD_NUM_ITERS \
    --md_lr $PROJECTION_MD_LR \
    $rescale_projected_latent_norm \
    --noise_std_scaling_factor $DIFF_STEP_NOISE_STD_SCALING_FACTOR \
    --coupling_grad_scale $COUPLING_GRAD_SCALE \
    --coupling_var "$COUPLING_VAR" \
    --coupling_fn_rot $COUPLING_FN_ROT \
    $enable_projection \
    $enable_coupling \
    --num_samples_to_generate $NUM_SAMPLES_TO_GENERATE \
    --image_classifier_config "$IMAGE_CLASSIFIER_CONFIG" \
    --image_classifier_weights "$IMAGE_CLASSIFIER_WEIGHTS" \
    --outputs_dir "$OUTPUTS_DIR" 2>&1 | tee -a "$log_file"  # pipe outputs & err to log file too


end_time=$(date +"%Y-%m-%d %H:%M:%S")
end_sec=$(date +%s)
echo "Script ended at: $end_time" | tee -a "$log_file"

# Calculate duration in different formats
duration_sec=$((end_sec - start_sec))
duration_min=$(echo "scale=2; $duration_sec / 60" | bc)
duration_hr=$(echo "scale=2; $duration_sec / 3600" | bc)

echo "Total duration: " | tee -a "$log_file"
echo "- In hours: ${duration_hr} hours" | tee -a "$log_file"
echo "- In minutes: ${duration_min} minutes" | tee -a "$log_file"
echo "- In seconds: ${duration_sec} seconds" | tee -a "$log_file"
