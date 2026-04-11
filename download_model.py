"""
Pre-downloads SPECTER2 model and proximity adapter into the HuggingFace cache.
Run once at Docker build time so the model is baked into the image.
"""
from transformers import AutoTokenizer
from adapters import AutoAdapterModel

model_name = "allenai/specter2_base"
adapter_name = "allenai/specter2"

print("Downloading tokenizer...")
AutoTokenizer.from_pretrained(model_name)

print("Downloading model...")
model = AutoAdapterModel.from_pretrained(model_name)

print("Downloading proximity adapter...")
model.load_adapter(adapter_name, source="hf", load_as="proximity", set_active=True)

print("Done.")
