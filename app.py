"""Run with: streamlit run app.py"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from bass_spot.config import Experiment
from bass_spot.geometry import make_room, valid_cells, validate_point
from bass_spot.models import Predictor
from bass_spot.optimize import (
    benchmark_placement,
    candidates_for,
    default_listeners,
    listening_points,
)
from bass_spot.solver import AcousticSolver, log_response

st.set_page_config(page_title="The Bass Spot", page_icon="🔊", layout="wide")
st.title("The Bass Spot")
st.write("Explore how speaker position changes bass in a simplified 2D room.")
st.caption("Relative simulation response • rigid walls • one idealised speaker • uncalibrated levels")


@st.cache_resource
def load_predictor(path, modified):
    torch.set_num_threads(2)
    return Predictor(path)


@st.cache_data(max_entries=16)
def simulate(mask_bytes, shape, parameters, kind, physics, point, frequency):
    from bass_spot.geometry import Room

    room = Room(np.frombuffer(mask_bytes, dtype=bool).reshape(shape), kind, parameters)
    return log_response(AcousticSolver(room, physics).solve(point, frequency), physics)


checkpoints = sorted(str(path) for path in Path("runs").glob("*/*/best.pt"))
with st.sidebar:
    st.header("Room and speaker")
    checkpoint = st.selectbox("Trained model", ["Simulator only", *checkpoints])
    predictor = None
    if checkpoint != "Simulator only":
        predictor = load_predictor(checkpoint, Path(checkpoint).stat().st_mtime_ns)
    config = predictor.config if predictor else Experiment()
    physics = config.physics
    kind = st.selectbox("Room shape", ["Rectangle", "L-shaped"])
    width = st.slider("Room width (m)", 3.5, float(physics.width), 5.0, 0.125)
    height = st.slider("Room height (m)", 3.0, float(physics.height), 4.0, 0.125)
    cut_x = cut_y = 0.0
    if kind == "L-shaped":
        cut_fraction = st.slider("Corner cutout: width fraction", 0.20, 0.72, 0.35, 0.01)
        cut_height_fraction = st.slider("Corner cutout: height fraction", 0.20, 0.72, 0.35, 0.01)
        cut_x, cut_y = width * cut_fraction, height * cut_height_fraction
        if cut_fraction > 0.45 or cut_height_fraction > 0.45:
            st.warning("This cutout is beyond the model's training range. Check simulator results.")
    room = make_room(physics, width, height, cut_x, cut_y)
    st.caption(f"Simulated dimensions: {room.parameters['width']:.3f} × {room.parameters['height']:.3f} m")
    sx = st.slider("Speaker x (m)", 0.25, float(width - 0.25), min(1.0, width - 0.25), 0.05)
    sy = st.slider("Speaker y (m)", 0.25, float(height - 0.25), min(1.0, height - 0.25), 0.05)
    frequency = st.select_slider("Frequency (Hz)", options=list(physics.frequencies), value=90.0)
    st.header("Listening area")
    default = default_listeners(room, physics).mean(axis=0)
    lx = st.number_input(
        "Sofa centre x (m)", min_value=0.0, max_value=float(width), value=float(default[0]), step=0.1
    )
    ly = st.number_input(
        "Sofa centre y (m)", min_value=0.0, max_value=float(height), value=float(default[1]), step=0.1
    )
    sofa_width = st.slider("Sofa width (m)", 0.5, 2.0, 1.0, 0.1)
    search = st.button("Find speaker position", type="primary", disabled=predictor is None)
    if predictor is None:
        st.info("Choose a trained model to compare predictions and search speaker positions.")

point = (sx, sy)
try:
    validate_point(room, physics, point)
    y, x = int(sy / physics.dy), int(sx / physics.dx)
    if not any(np.array_equal(cell, (y, x)) for cell in valid_cells(room, physics)):
        st.warning("Speaker is closer to the wall than training sources; treat predictions cautiously.")
    listeners = listening_points(room, physics, (lx, ly), sofa_width)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

signature = (room.geometry_hash, checkpoint, tuple(listeners.ravel()), sx, sy)
if search:
    with st.spinner("Ranking candidate positions and checking them with the simulator…"):
        candidates = candidates_for(room, physics, listeners, config.candidates, np.random.default_rng(2026))
        result = benchmark_placement(room, listeners, predictor, candidates)
        st.session_state["recommendation"] = (signature, result)

stored = st.session_state.get("recommendation")
recommendation = stored[1] if stored and stored[0] == signature else None
if recommendation:
    point = recommendation["candidates"][recommendation["selected_index"]]
    st.success(f"Suggested speaker position: ({point[0]:.2f}, {point[1]:.2f}) m")
    cols = st.columns(3)
    cols[0].metric("Simulator-verified score ↓", f"{recommendation['selected_score']:.2f}")
    cols[1].metric("Best candidate score ↓", f"{recommendation['oracle_score']:.2f}")
    cols[2].metric("Regret ↓", f"{recommendation['regret']:.2f}")
    with st.expander("Candidate rankings and measured runtime"):
        st.dataframe(
            {
                "Position (m)": [str(np.round(p, 2)) for p in recommendation["candidates"]],
                "Predicted score": recommendation["predicted_scores"],
                "Simulator score": recommendation["simulator_scores"],
            }
        )
        st.write({key: value for key, value in recommendation.items() if key.endswith("seconds")})

truth = simulate(
    room.mask.tobytes(), room.mask.shape, room.parameters, room.kind, physics, tuple(point), frequency
)
arrays, titles = [truth], ["Numerical simulator"]
if predictor:
    prediction = predictor.predict(room, [point], [frequency])[0, 0]
    arrays = [prediction, truth, np.abs(prediction - truth)]
    titles = ["Neural prediction", "Numerical simulator", "Absolute error"]
    st.caption(f"Mean absolute error in this room: {np.abs(prediction - truth)[room.mask].mean():.2f} dB")
low, high = np.quantile(truth[room.mask], [0.02, 0.98])
fig, axes = plt.subplots(1, len(arrays), figsize=(5 * len(arrays), 4), squeeze=False, constrained_layout=True)
for index, (ax, array, title) in enumerate(zip(axes[0], arrays, titles)):
    kwargs = dict(vmin=0, cmap="magma") if index == 2 else dict(vmin=low, vmax=high, cmap="viridis")
    im = ax.imshow(
        np.ma.masked_where(~room.mask, array),
        origin="lower",
        extent=(0, physics.width, 0, physics.height),
        **kwargs,
    )
    ax.scatter(*point, color="red", marker="*", s=90, label="Speaker")
    ax.scatter(listeners[:, 0], listeners[:, 1], color="white", edgecolor="black", s=20, label="Listeners")
    ax.set(title=title, xlabel="x (m)", ylabel="y (m)")
    ax.legend(loc="upper right", fontsize=7)
    fig.colorbar(im, ax=ax, label="Relative log response (dB)")
st.pyplot(fig)
plt.close(fig)
st.caption(
    "Placement balances variation across the sofa and weak mean response over all four frequencies. "
    "The best candidate is the best among the sampled positions, rather than a global optimum."
)
