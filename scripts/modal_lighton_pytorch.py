import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor


def select_device() -> tuple[str, torch.dtype]:
    """
    Select the appropriate device and dtype for the model.
    Priority: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
    """
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float32  # MPS performs better with float32
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16  # CUDA supports bfloat16 for better efficiency
    else:
        device = "cpu"
        dtype = torch.float32
    
    return device, dtype


def load_model(model_name: str = "lightonai/LightOnOCR-2-1B"):
    """Load the OCR model with appropriate dtype and device."""
    device, dtype = select_device()
    
    print(f"Using device: {device} with dtype: {dtype}")
    
    model = LightOnOcrForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype
    ).to(device)
    
    # Set model to evaluation mode for inference
    model.eval()
    
    processor = LightOnOcrProcessor.from_pretrained(model_name)
    
    return model, processor, device, dtype


def extract_text_from_image(
    model: LightOnOcrForConditionalGeneration,
    processor: LightOnOcrProcessor,
    image_url: str,
    device: str,
    dtype: torch.dtype,
    max_new_tokens: int = 1024
) -> str:
    """
    Extract text from an image using the OCR model.
    
    Args:
        model: The loaded OCR model
        processor: The processor for the model
        image_url: URL of the image to process
        device: Device to run inference on
        dtype: Data type for model computation
        max_new_tokens: Maximum tokens to generate
    
    Returns:
        Extracted text from the image
    """
    conversation = [
        {
            "role": "user",
            "content": [{"type": "image", "url": image_url}]
        }
    ]
    
    # Prepare inputs
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    
    # Move inputs to device with appropriate dtype
    inputs = {
        k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
        for k, v in inputs.items()
    }
    
    # Inference with no_grad for memory efficiency
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Use greedy decoding for consistency
        )
    
    # Extract generated text (excluding input tokens)
    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    output_text = processor.decode(generated_ids, skip_special_tokens=True)
    
    return output_text


def main():
    """Main function to demonstrate OCR extraction."""
    # Initialize model and processor
    model, processor, device, dtype = load_model()
    
    # Example image URL
    url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_ocr/resolve/main/SROIE-receipt.jpeg"
    
    # Extract text from image
    print("Extracting text from image...")
    extracted_text = extract_text_from_image(
        model,
        processor,
        url,
        device,
        dtype,
        max_new_tokens=1024
    )
    
    print("\nExtracted Text:")
    print(extracted_text)


if __name__ == "__main__":
    main()
