import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.errors import RepositoryNotFoundError

# -----------------------------
# CONFIG
# -----------------------------
SPACE_ID = "BujjiProjectPrep/Tourist-Customer-Prediction-0612"
SPACE_TYPE = "space"
SPACE_SDK = "streamlit"  # <- required now

LOCAL_DEPLOYMENT_FOLDER = "tourist_customer_prediction/model_deployment"

def get_hf_api():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "In Colab, run:  import os; os.environ['HF_TOKEN'] = 'your_hf_token_here'"
        )
    api = HfApi(token=hf_token)
    return api

def create_or_get_space(api: HfApi):
    try:
        api.repo_info(repo_id=SPACE_ID, repo_type=SPACE_TYPE)
        print(f"✅ Space '{SPACE_ID}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"ℹ️ Space '{SPACE_ID}' not found. Creating a new one...")
        create_repo(
            repo_id=SPACE_ID,
            repo_type=SPACE_TYPE,
            space_sdk=SPACE_SDK,   # <- IMPORTANT
            private=False,
        )
        print(f"✅ Space '{SPACE_ID}' created successfully.")

def upload_deployment_files(api: HfApi):
    print(f"☁️ Uploading deployment files from '{LOCAL_DEPLOYMENT_FOLDER}' to HF Space...")

    api.upload_folder(
        folder_path=LOCAL_DEPLOYMENT_FOLDER,
        repo_id=SPACE_ID,
        repo_type=SPACE_TYPE,
    )

    print("🎉 Deployment files uploaded successfully.")
    print(f"🔗 Space URL: https://huggingface.co/spaces/{SPACE_ID}")

def main():
    api = get_hf_api()
    create_or_get_space(api)
    upload_deployment_files(api)

if __name__ == "__main__":
    main()
