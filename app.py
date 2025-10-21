import argparse
import logging
import torch
from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
# ----------------------------
# Argument parsing
# ----------------------------
parser = argparse.ArgumentParser(description="Run the Flask app for HTML generation using an LLM.")
parser.add_argument(
    "--model_name",
    type=str,
    default="Qwen/Qwen3-4B-Thinking-2507",
    help="The Hugging Face model to load."
)
parser.add_argument(
    "--system_prompt",
    type=str,
    default=(
        "You are a professional web developer. The user provides content (text/markdown), and you generate a complete, "
        "multipage, modern, responsive website using only HTML, CSS, and minimal vanilla JavaScript. "
        "Your output must be a valid JSON object where each key is a relative file path (e.g., 'home.html', 'features.html') "
        "and each value is the full, self-contained HTML source code for that page. "
        "Each HTML file must:\n"
        "- Be a complete, valid HTML5 document.\n"
        "- Include inline CSS in a <style> tag (no external files).\n"
        "- Use minimal, clean JavaScript only when necessary.\n"
        "- Support being embedded in an iframe: all internal links must use 'target=\"_self\"' to stay inside the iframe.\n"
        "- Include a script that posts a 'message' to parent window on load and resize to report its scrollHeight:\n"
        "  window.parent.postMessage({ height: document.body.scrollHeight }, '*');\n"
        "- NOT include any markdown, explanations, or text outside the JSON.\n"
        "Example output: {\"home.html\": \"<!DOCTYPE html>...\", \"features.html\": \"<!DOCTYPE html>...\"}"
    ),
    help="Custom system prompt for the model’s behavior."
)

args = parser.parse_args()
MODEL_NAME = args.model_name
SYSTEM_PROMPT = args.system_prompt

# ----------------------------
# Flask app setup
# ----------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ----------------------------
# Model loading
# ----------------------------
app.logger.info(f"Loading model: {MODEL_NAME}")



# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def index():
    return render_template("index.html")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config={
        "load_in_4bit": True,
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.float16
    }
)
# ----------------------------
# Generation helper
# ----------------------------
def generate_html_from_prompt(prompt_text: str, max_new_tokens: int = 32768, do_sample=True, temperature=0.7, top_p=0.95):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    try:
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        output_ids = generated_ids[0][len(inputs["input_ids"][0]):].tolist()
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        # Optional: remove internal reasoning traces
        if "</think>" in content:
            content = content.split("</think>", 1)[1].strip()

        return content
    finally:
        # Clean up
        del inputs
        del generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# ----------------------------
# API endpoint
# ----------------------------
@app.route("/generate", methods=["POST"])
def generate_route():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "").strip()
    strip_scripts = bool(data.get("strip_scripts", False))

    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    app.logger.info(f"Received generation request (len={len(prompt)})")

    try:
        generated = generate_html_from_prompt(prompt_text=prompt)

        # Optional: remove <script> tags if requested
        if strip_scripts:
            import re
            generated = re.sub(r"<script[\s\S]*?</script>", "", generated, flags=re.IGNORECASE)

        return jsonify({"html": generated})
    except Exception as e:
        app.logger.exception("Generation failed")
        return jsonify({"error": str(e)}), 500


# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    app.logger.info(f"Running with model: {MODEL_NAME}")
    app.logger.info(f"System prompt: {SYSTEM_PROMPT[:100]}{'...' if len(SYSTEM_PROMPT) > 100 else ''}")
    app.run(host="0.0.0.0", port=5000, debug=False)

