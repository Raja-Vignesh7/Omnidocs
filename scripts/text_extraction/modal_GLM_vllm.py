import modal

IMAGE = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libglib2.0-0", "libgl1", "libglx-mesa0")
    .run_commands(
        "pip install uv",
        "uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cu124",
        "uv pip install --system -U vllm --extra-index-url https://wheels.vllm.ai/nightly",
        "uv pip install --system git+https://github.com/huggingface/transformers.git",
        "uv pip install --system pillow pypdfium2 requests",
    )
)

app = modal.App("glm-ocr-vllm")

@app.function(
    image=IMAGE,
    gpu="A10G:1",
    timeout=600,
)
def run_glm_ocr(image_url: str):
    import base64
    import socket
    import subprocess
    import time
    from io import BytesIO

    import requests
    from PIL import Image

    model = "zai-org/GLM-OCR"

    #  Start vLLM server
    server = subprocess.Popen(
        [
            "vllm",
            "serve",
            model,
            "--allowed-local-media-path",
            "/",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--limit-mm-per-prompt",
            '{"image": 1}',
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    def wait_for_server(host="localhost", port=8000, timeout=1200):
        start = time.time()
        while time.time() - start < timeout:
            # If server crashed, exit early
            if server.poll() is not None:
                stdout, stderr = server.communicate()
                print("STDOUT:", stdout.decode())
                print("STDERR:", stderr.decode())
                raise RuntimeError("vLLM server failed to start.")

            try:
                with socket.create_connection((host, port), timeout=2):
                    return
            except OSError:
                time.sleep(2)

        raise RuntimeError("Timed out waiting for vLLM server.")

    wait_for_server()
    
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    {"type": "text", "text": "Transcribe this document."},
                ],
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.2,
        "top_p": 0.9,
    }
    
    result = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json=payload,
        timeout=300,
    )
    
    result.raise_for_status()
    
    output_text = result.json()["choices"][0]["message"]["content"]
    
    server.terminate()
    server.wait()
    
    return output_text

@app.local_entrypoint()
def main():
    # target_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_ocr/resolve/main/SROIE-receipt.jpeg"
    target_url = (
        "https://media2.dev.to/dynamic/image/width=1080,height=1080,"
        "fit=cover,gravity=auto,format=auto/"
        "https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles"
        "%2Frgkpspl9cq1svsjjtr9f.webp"
    )
    result = run_glm_ocr.remote(target_url)

    print("\n" + "=" * 50)
    print("OCR RESULT:")
    print("=" * 50)
    print(result)