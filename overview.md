# flux-portrait — AI Photo Generation Pipeline

## Project Overview

A self-hosted pipeline for training a personal LoRA (Low-Rank Adaptation) model on a small set of face photos, then using that model to generate photorealistic images of a specific person in any scene or context — dating app profiles, LinkedIn headshots, lifestyle photos, and more.

The system combines a large pre-trained text-to-image base model (which understands the world: beaches, boats, cities, lighting, etc.) with a lightweight personal adapter (the LoRA) that teaches it what a specific person looks like. At generation time, the two are combined: the user prompts with a trigger word + scene description and gets a realistic photo of themselves in that scene.

---

## Background & Motivation

AI-generated dating and profile photos are a growing category. Services like DatePhotos.ai, Narkis.ai, and MatchPhotos.io all offer this as a paid product ($27–49 per batch), but they:

- Require uploading personal face photos to third-party servers
- Deliver a fixed batch of photos with limited prompt control
- Charge per use, with no reuse or ongoing access
- Use the same underlying technology that is now fully open-source

This project replicates that pipeline locally (or on Google Colab), using identical technology — Flux.1 Dev as the base model and Kohya_ss for LoRA training — giving the user:

- **Full privacy**: photos never leave the user's machine (or private Colab session)
- **Unlimited generation**: run as many prompts as needed after training
- **Full prompt control**: any scene, lighting, outfit, background, platform format
- **Reusability**: one trained LoRA works for dating apps, LinkedIn, personal branding, etc.
- **Cost efficiency**: ~$3–5 in Google Colab compute per training run vs. $29+ per service use

The user already has experience with Google Colab (used for a financial investments project) and is comfortable with a technical setup process. Primary use cases are Hinge/dating app profile photos and LinkedIn headshots, with specific interest in generating photos in real-world scenes (e.g., on a boat, at a marina, in a city).

---

## Goals

1. Train a Flux.1 LoRA on 15–30 personal face photos with minimal manual configuration
2. Generate photorealistic, natural-looking output images suitable for dating apps and LinkedIn
3. Support custom scene prompts (boat, beach, city, office, restaurant, outdoors, etc.)
4. Run training on Google Colab (A100 or L4) using existing credits, or locally if GPU permits
5. Provide a simple CLI or notebook interface for both training and generation
6. Output images at appropriate resolutions for each platform (Hinge, LinkedIn, etc.)

---

## Technical Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER INPUT                           │
│  15-30 face photos (JPG/PNG, varied angles/lighting)    │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              PHASE 1: DATA PREPARATION                  │
│  - Auto-crop to face-centred 1024x1024                  │
│  - Auto-caption using Florence-2 or BLIP-2              │
│  - Inject trigger word into captions (e.g. "ohwx man")  │
│  - Quality filter: blur detection, duplicate removal     │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              PHASE 2: LORA TRAINING                     │
│  Base model : Flux.1 Dev (open weights, HuggingFace)    │
│  Trainer    : Kohya_ss (sd-scripts, flux branch)        │
│  Output     : .safetensors LoRA file (~50-200MB)        │
│  Runtime    : ~20-40 min on Colab A100                  │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              PHASE 3: GENERATION                        │
│  Runtime    : ComfyUI (local) or diffusers (Colab)      │
│  Input      : trigger word + scene prompt               │
│  Output     : 1024x1024 PNG, batch of N images          │
│  Upscale    : optional 2x/4x with RealESRGAN            │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              PHASE 4: POST-PROCESSING                   │
│  - Resize/crop to platform-specific dimensions          │
│  - Optional: face restoration with CodeFormer/GFPGAN    │
│  - Output organised by use case (dating / linkedin)     │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Tool | Version / Notes |
|---|---|---|
| Base model | Flux.1 Dev | Open weights via HuggingFace (`black-forest-labs/FLUX.1-dev`) |
| LoRA trainer | Kohya_ss sd-scripts | Flux branch — `kohya-ss/sd-scripts` |
| Inference (local) | ComfyUI | Latest stable, with ComfyUI-Manager |
| Inference (Colab) | `diffusers` + `peft` | HuggingFace pipeline |
| Auto-captioning | Florence-2 or BLIP-2 | Via `transformers` |
| Face crop/prep | `insightface` + `opencv-python` | RetinaFace for detection |
| Upscaling | RealESRGAN | `realesrgan` Python package |
| Face restoration | CodeFormer | Optional post-processing step |
| Environment | Python 3.10 | Conda or venv |
| GPU (Colab) | A100 40GB (preferred) or L4 24GB | Colab Pro/Pro+ |
| GPU (local) | NVIDIA 16GB+ VRAM | RTX 3080 Ti / 4080 / 4090 |
| Package manager | `pip` + `conda` | |
| IDE | VS Code | With Python and Jupyter extensions |

---

## Project Structure

```
flux-portrait/
│
├── README.md                    # This file
│
├── data/
│   ├── raw/                     # Original input photos (gitignored)
│   ├── processed/               # Cropped + captioned training images
│   └── captions/                # .txt caption files (one per image)
│
├── models/
│   ├── base/                    # Flux.1 Dev weights (downloaded, gitignored)
│   └── lora/                    # Trained LoRA .safetensors output files
│
├── output/
│   ├── dating/                  # Generated photos for dating apps
│   ├── linkedin/                # Generated photos for LinkedIn
│   └── raw/                     # Unprocessed generation output
│
├── scripts/
│   ├── prepare_data.py          # Face crop, caption, quality filter
│   ├── train_lora.py            # Kohya_ss training launcher
│   ├── generate.py              # CLI generation script
│   ├── postprocess.py           # Upscale, face restore, resize
│   └── utils.py                 # Shared helpers
│
├── configs/
│   ├── training_config.toml     # Kohya_ss training parameters
│   ├── flux_lora_default.json   # Default LoRA architecture config
│   └── prompt_templates.yaml    # Prompt templates by use case
│
├── notebooks/
│   ├── 01_prepare_data.ipynb    # Colab-ready data prep notebook
│   ├── 02_train_lora.ipynb      # Colab-ready training notebook
│   └── 03_generate.ipynb        # Colab-ready generation notebook
│
├── comfyui_workflows/
│   └── flux_lora_basic.json     # ComfyUI workflow for local generation
│
├── requirements.txt
├── requirements_colab.txt       # Colab-specific dependencies
└── .env.example                 # HuggingFace token etc.
```

---

## Training Configuration (Key Parameters)

These are the recommended starting parameters for a face LoRA on Flux.1 Dev. Claude Code should use these as defaults in `configs/training_config.toml`:

```toml
[training]
pretrained_model_name_or_path = "black-forest-labs/FLUX.1-dev"
output_dir = "./models/lora"
output_name = "flux_portrait_v1"

# Dataset
train_data_dir = "./data/processed"
resolution = "1024,1024"
train_batch_size = 1
gradient_accumulation_steps = 4

# Training duration
max_train_steps = 1500          # Start here; 1000-2000 is typical
save_every_n_steps = 250        # Save checkpoints to evaluate quality

# LoRA architecture
network_module = "networks.lora_flux"
network_dim = 16                # Rank; 8-32 is typical for faces
network_alpha = 16              # Usually equal to dim
network_train_unet_only = true  # Flux: train UNet only

# Learning rate
learning_rate = 5e-5
unet_lr = 5e-5
lr_scheduler = "cosine_with_restarts"
lr_warmup_steps = 100

# Precision / memory
mixed_precision = "bf16"
save_precision = "bf16"
xformers = false                # Not needed for Flux
sdpa = true                     # Use scaled dot-product attention

# Trigger word (injected into all captions)
# User should pick a rare/unique word e.g. "ohwx man"
# This word is used at inference time to invoke the learned face
```

---

## Data Preparation Requirements

### Input photos (user provides)
- **Quantity**: 15–30 photos minimum; 25 is the sweet spot
- **Variety required**:
  - Multiple angles: front-facing, 3/4 left, 3/4 right, slight profile
  - Multiple lighting conditions: natural daylight, indoor, overcast, golden hour
  - Multiple expressions: neutral, smiling, candid/laughing
  - With and without sunglasses, hats (include some of each)
  - Different backgrounds (outdoors, indoor, varied colours)
  - Some full-face close-ups AND some mid-body shots
- **Avoid**: heavy filters, heavily edited photos, group photos, low resolution (<512px face area)

### Auto-processing (scripts handle this)
- Face detection and centre-crop to 1024x1024
- Blur score filter (reject if Laplacian variance < threshold)
- BLIP-2 or Florence-2 auto-captioning
- Trigger word injection: all captions prepended with `"a photo of ohwx man, "`
- Output: image + matching `.txt` caption file pairs

---

## Prompt Templates by Use Case

Stored in `configs/prompt_templates.yaml`. Examples:

```yaml
dating_casual:
  - "a photo of ohwx man sitting on a boat deck at golden hour, relaxed smile, navy casual shirt, candid"
  - "a photo of ohwx man at an outdoor café, coffee on table, warm afternoon light, natural expression"
  - "a photo of ohwx man hiking on a mountain trail, scenic background, casual outdoors clothing"
  - "a photo of ohwx man laughing at a rooftop bar, city lights behind, evening, social setting"

dating_active:
  - "a photo of ohwx man on a sailboat, hands on wheel, ocean background, confident expression"
  - "a photo of ohwx man kayaking on a lake, action shot, summer, athletic"
  - "a photo of ohwx man at a beach, casual summer outfit, natural candid moment"

linkedin_professional:
  - "a professional headshot of ohwx man, neutral background, business casual, soft studio lighting, direct eye contact"
  - "a photo of ohwx man in a modern office environment, confident posture, professional attire"
  - "a photo of ohwx man presenting at a whiteboard, engaged expression, business setting"

negative_prompt: "cartoon, illustration, painting, drawing, anime, render, deformed, ugly,
  blurry, low quality, watermark, text, logo, bad anatomy, extra fingers,
  mutated hands, poorly drawn face, bad proportions, gross proportions"
```

---

## Platform Output Specifications

```yaml
hinge:
  resolution: 1080x1350      # 4:5 portrait (recommended)
  fallback: 1080x1080        # Square also accepted
  format: JPG
  max_size_mb: 10

tinder:
  resolution: 640x800        # 4:5 portrait
  format: JPG

bumble:
  resolution: 1080x1080      # Square preferred
  format: JPG

linkedin:
  resolution: 400x400        # Profile photo (square)
  format: JPG
  note: "Use professional headshot templates only"
```

---

## Environment Setup

### Option A — Google Colab (Recommended)

1. Use Colab Pro or Pro+ with A100 or L4 runtime
2. Run notebooks in order: `01_prepare_data.ipynb` → `02_train_lora.ipynb` → `03_generate.ipynb`
3. Mount Google Drive for persistent model storage across sessions
4. Requires a HuggingFace account and access token for Flux.1 Dev (free, gated model)
5. Estimated Colab compute cost: ~$3–8 for a full training run on A100

### Option B — Local (VS Code)

Requirements:
- NVIDIA GPU with 16GB+ VRAM (24GB recommended for Flux)
- CUDA 12.1+
- Python 3.10 via conda
- ~60GB disk space (Flux.1 Dev weights are ~24GB)

Setup:
```bash
conda create -n personal-lora python=3.10
conda activate personal-lora
pip install -r requirements.txt
```

### HuggingFace Access

Flux.1 Dev is a gated model. The user must:
1. Create a free account at huggingface.co
2. Accept the Flux.1 Dev licence at `huggingface.co/black-forest-labs/FLUX.1-dev`
3. Generate an access token (Settings → Access Tokens)
4. Set `HF_TOKEN=your_token_here` in `.env`

---

## Implementation Phases for Claude Code

### Phase 1 — Project scaffold & environment
- Create full directory structure as specified above
- Write `requirements.txt` and `requirements_colab.txt`
- Write `.env.example` with all required environment variables
- Write base `utils.py` with shared helpers (logging, path management, config loading)

### Phase 2 — Data preparation script
- `scripts/prepare_data.py`:
  - Accept input directory of raw photos
  - Detect faces with InsightFace RetinaFace
  - Crop and pad to 1024x1024 centred on face
  - Score sharpness (Laplacian variance), reject below threshold
  - Remove near-duplicate images (perceptual hash)
  - Run BLIP-2 or Florence-2 captioning
  - Inject trigger word into all captions
  - Write output image + `.txt` pairs to `data/processed/`
  - Print summary: N accepted, N rejected, reasons

### Phase 3 — Training launcher
- `scripts/train_lora.py`:
  - Install/verify Kohya_ss sd-scripts (flux branch) as a submodule or dependency
  - Load `configs/training_config.toml`
  - Validate that training data exists and is correctly formatted
  - Launch Kohya_ss training subprocess with correct args
  - Monitor output and report progress
  - Save final LoRA to `models/lora/`

### Phase 4 — Generation script
- `scripts/generate.py`:
  - Accept: LoRA path, prompt, negative prompt, number of images, output use case
  - Load Flux.1 Dev + LoRA via `diffusers` + `peft`
  - Run inference with configurable steps (28 default) and guidance scale (3.5)
  - Save outputs to `output/{use_case}/`
  - Optional: pass through RealESRGAN upscaler

### Phase 5 — Colab notebooks
- Mirror each script as a self-contained Colab notebook
- Include Drive mount, model download, and pip install cells at top
- Make each notebook runnable top-to-bottom with no external dependencies

### Phase 6 — ComfyUI workflow (optional local UI)
- Export a `comfyui_workflows/flux_lora_basic.json` workflow
- Nodes: Load Checkpoint → Load LoRA → CLIP Text Encode (positive + negative) → KSampler → VAE Decode → Save Image
- Pre-wired for Flux.1 Dev + user's LoRA path

---

## Key Decisions & Rationale

**Why Flux.1 Dev over SDXL?**
Flux produces significantly better photorealism, more accurate anatomy, and better face consistency than SDXL. The larger memory footprint is acceptable given Colab A100 access.

**Why Kohya_ss over SimpleTuner or other trainers?**
Kohya_ss is the most battle-tested LoRA trainer, has active Flux support, and has the largest community of face LoRA examples to reference. SimpleTuner is a valid alternative if Kohya_ss setup proves difficult.

**Why a trigger word approach?**
Using a rare trigger word (e.g. `ohwx man`) prevents the LoRA from bleeding into general prompts and gives precise control. It's the industry-standard approach for identity LoRAs.

**Why not use a face-swap approach instead?**
Face-swap tools (ReActor, InsightFace swap) paste a face onto a generated body and often produce lighting/skin-tone mismatches. LoRA training generates the person holistically and handles lighting, shadows, and expression naturally.

---

## Out of Scope (for now)

- Multi-person LoRAs
- Video generation
- Real-time inference API / web UI
- Automatic A/B testing of generated photos
- Integration with dating app APIs

---

## References

- Flux.1 Dev model: `https://huggingface.co/black-forest-labs/FLUX.1-dev`
- Kohya_ss sd-scripts: `https://github.com/kohya-ss/sd-scripts`
- ComfyUI: `https://github.com/comfyanonymous/ComfyUI`
- Florence-2 captioner: `https://huggingface.co/microsoft/Florence-2-large`
- RealESRGAN: `https://github.com/xinntao/Real-ESRGAN`