# Run this in Python (in your api folder)
from huggingface_hub import HfApi, create_repo

# Replace with your HuggingFace username
USERNAME = "your-username"  # e.g., "johnsmith"
REPO_NAME = f"{USERNAME}/digital-empathy-assistant"

# Create the repository
api = HfApi()
api.create_repo(repo_id=REPO_NAME, private=False, exist_ok=True)

print(f"✅ Repository created: https://huggingface.co/{REPO_NAME}")

# Upload your model folder
api.upload_folder(
    folder_path="../models/final_deberta_multitask",
    repo_id=REPO_NAME,
    repo_type="model"
)

print(f"✅ Model uploaded to: https://huggingface.co/{REPO_NAME}")