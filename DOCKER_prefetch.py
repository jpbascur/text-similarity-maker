"""
Pre-downloads SPECTER2 model and proximity adapter into the HuggingFace cache.
Run once at Docker build time so the model is baked into the image.
"""
from importlib.metadata import version

transformers_version = version("transformers")
major_minor = tuple(int(part) for part in transformers_version.split(".")[:2])
if major_minor < (4, 40) or major_minor >= (4, 52):
    raise RuntimeError(
        f"Unsupported transformers {transformers_version}. "
        "Install requirements.txt so transformers stays in the supported >=4.40,<4.52 range."
    )

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
