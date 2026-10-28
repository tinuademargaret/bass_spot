# Streamlit Community Cloud

Deploy `tinuademargaret/bass_spot`, branch `main`, entrypoint `deploy/app.py`, with Python 3.13.

The requirements beside this entrypoint take priority over the root uv.lock and install CPU-only PyTorch. The app includes the baseline and boundary checkpoints from seed 11 of the full study. Other local checkpoints under runs/ remain available during local use.
