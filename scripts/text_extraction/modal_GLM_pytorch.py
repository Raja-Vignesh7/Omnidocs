import modal


IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "libglib2.0-0", "libgl1", "libglx-mesa0")
    .run_commands("pip install uv")
    # Install the latest transformers (v5 candidate) + dependencies
    .run_commands(
        "uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cu124",
        "uv pip install --system pillow pypdfium2",
        "uv pip install --system requests",
        "uv pip install --system transformers>=5.0.0",  # Ensures v5 support
        "uv pip install --system huggingface_hub[hf_transfer] accelerate",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("glm-ocr-experiment")

volume = modal.Volume.from_name("ocr-cache", create_if_missing=True)

app = modal.App("glm-ocr-experiment")
volume = modal.Volume.from_name("ocr-cache", create_if_missing=True)


@app.function(
    image=IMAGE,
    gpu="A10G:1",  # A10G GPU with 1 instance for cost-effective inference
    volumes={"/data": volume},
    timeout=600,  # 10min timeout for long generations
)
def run_glm_ocr(image_url: str) -> str:
    """
    Remote OCR/Text Extraction Function.

    Loads GLM-OCR model, downloads image from URL, processes via chat template,
    generates extracted text.

    Args:
        image_url (str): Public URL to RGB image (e.g., JPG/PNG).

    Returns:
        str: Extracted text from image.

    Raises:
        requests.RequestException: If image download fails.
        RuntimeError: Model loading/inference errors.
    """

    from io import BytesIO

    import requests
    import torch
    from PIL import Image
    
    from transformers import AutoProcessor, AutoModelForImageTextToText
    # Removed duplicate torch import

    model_id = "zai-org/GLM-OCR"
    """
    GLM-OCR: Multimodal model for document text recognition.
    Supports image+text chat inputs.
    """
    device = "cuda"
    dtype = torch.bfloat16  # Efficient for A10G; fallback to auto

    print(f"Loading {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        pretrained_model_name_or_path=model_id,
        torch_dtype="auto",  # Respects dtype
        device_map="auto",   # Auto-distributes across GPU(s)
    )
    # Caches in /data volume for faster subsequent calls
    # Download and preprocess image
    response = requests.get(image_url)
    response.raise_for_status()  # Raise on HTTP errors
    image = Image.open(BytesIO(response.content)).convert("RGB")
    """
    Convert to RGB: Standardizes for model input.
    Resizes/crops handled by processor.
    """

    # Prepare multi-turn conversation for GLM chat format
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Text Recognition:"},
            ],
        }
    ]
    """
    GLM-OCR specific: Image as separate dict entry, text prompt follows.
    Enables accurate document text extraction.
    """

    # Tokenize conversation with chat template
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    inputs.pop("token_type_ids", None)  # Not needed for GLM
    
    # Ensure tensors on GPU (device_map handles dtype)
    for k, v in inputs.items():
        if v.is_floating_point():
            inputs[k] = v.to(dtype=dtype)

    print("Generating text...")
    with torch.inference_mode():  # Memory-efficient, no gradients
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=8192,  # Long docs
            do_sample=False,      # Greedy for consistency
            pad_token_id=processor.tokenizer.eos_token_id,
        )
        # Extract generated tokens (post input_ids)
        new_tokens = generated_ids[0][inputs["input_ids"].shape[1] :]
        output_text = processor.decode(new_tokens, skip_special_tokens=True)

    return output_text
    
    # generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    # response = processor.decode(generated_ids, skip_special_tokens=True)
    # # print("Model response:", response)
    # return response[0]
    return output_text



@app.local_entrypoint()
def main():
    """
    Local Entrypoint for testing.

    Runs remote function with sample image URL and prints extracted text.
    Usage: python scripts/text_extraction/modal_GLM_pytorch.py
    """
    # Sample dev.to article image for testing
    image_url = (
        "https://media2.dev.to/dynamic/image/width=1080,height=1080,"
        "fit=cover,gravity=auto,format=auto/"
        "https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles"
        "%2Frgkpspl9cq1svsjjtr9f.webp"
    )
    result = run_glm_ocr.remote(image_url)
    print("Extracted text:", result)

    
    
    