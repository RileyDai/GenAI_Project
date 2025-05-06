import streamlit as st
import subprocess
import glob
import os
import yaml
from PIL import Image

st.title("VAE Model Image Generator")

script_path = st.text_input("Path to VAE script", "./run_vae_v1.py")
config_path = st.text_input("Path to config file", "./config/test_apples.yaml")

if st.button("Generate Images"):
    cmd = ["python", script_path, "--config", config_path]
    st.write("Executing command: `" + " ".join(cmd) + "`")
    with st.spinner("Running model..."):
        proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        st.error("Error occurred during execution:\n" + proc.stderr)
    else:
        st.success("Model finished running!")
        try:
            with open(config_path, "r") as cf:
                cfg = yaml.safe_load(cf)
            model_path = cfg.get("model_path")
        except Exception as e:
            st.error(f"Failed to read config file: {e}")
            model_path = None

        if not model_path:
            st.warning("model_path is not set in the config file. Please check your configuration.")
        else:
            model_dir = os.path.dirname(model_path)
            img_path = os.path.join(model_dir, "sample_epoch_inference.png")
            if os.path.isfile(img_path):
                img = Image.open(img_path)
                st.image(img, caption=os.path.basename(img_path), use_container_width=True)
            else:
                st.warning(f"Could not find sample_epoch_inference.png in {model_dir}.")
