import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Turbofan RUL Prediction — Digital Twin", layout="wide")

# -------------------------
# Model definition + loading (cached so it only loads once)
# -------------------------

class LSTMModel(nn.Module):
    def __init__(self, num_features, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, (hidden, cell) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction.squeeze()


@st.cache_resource
def load_everything():
    with open("config.pkl", "rb") as f:
        config = pickle.load(f)
    feature_cols = config["feature_cols"]
    sequence_length = config["sequence_length"]

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    model = LSTMModel(num_features=len(feature_cols))
    model.load_state_dict(torch.load("lstm_model.pth", map_location="cpu"))
    model.eval()

    test_df = pd.read_csv("test_engines_sample.csv")

    return model, scaler, feature_cols, sequence_length, test_df


model, scaler, feature_cols, SEQUENCE_LENGTH, test_df = load_everything()
engine_ids = sorted(test_df["unit_number"].unique().tolist())
RUL_CAP = 125

# -------------------------
# Header
# -------------------------

st.title("Turbofan Engine RUL Prediction — Deep Learning / PHM")
st.markdown(
    "A digital-twin style prognostics demo: an LSTM trained on NASA's C-MAPSS turbofan "
    "degradation dataset predicts **Remaining Useful Life (RUL)** — how many operating cycles "
    "an engine has left before failure — from multivariate sensor time-series data.\n\n"
    "[View the full project & code on GitHub](YOUR_GITHUB_LINK_HERE)"
)

st.divider()

# -------------------------
# Section: Problem statement
# -------------------------

with st.expander("What problem is this solving? (click to read)"):
    st.markdown(
        "In Prognostics and Health Management (PHM), predicting equipment failure before it "
        "happens allows maintenance to be scheduled proactively — avoiding both unnecessary "
        "early part replacement and dangerous late failure. This model estimates an aircraft "
        "engine's Remaining Useful Life from its recent sensor history (temperatures, pressures, "
        "speeds, etc.), using NASA's public C-MAPSS turbofan simulation dataset."
    )

st.divider()

# -------------------------
# Section: Engine Explorer
# -------------------------

st.header("🔧 Engine Explorer — Live Digital Twin")
st.markdown(
    "Pick a test engine (real historical data, never seen during training) to see how the "
    "model's RUL prediction evolves cycle-by-cycle — simulating a live health-monitoring dashboard."
)

col1, col2 = st.columns([1, 3])
with col1:
    selected_engine = st.selectbox("Select Engine ID", engine_ids)

engine_data = test_df[test_df["unit_number"] == selected_engine].sort_values("time_cycles")
features = engine_data[feature_cols].values
cycles = engine_data["time_cycles"].values

if len(features) < SEQUENCE_LENGTH:
    st.warning(f"Engine {selected_engine} has fewer than {SEQUENCE_LENGTH} cycles — not enough history to predict.")
else:
    predicted_ruls = []
    cycle_points = []
    with torch.no_grad():
        for i in range(SEQUENCE_LENGTH, len(features) + 1):
            window = features[i - SEQUENCE_LENGTH:i]
            window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
            pred = model(window_tensor)
            predicted_ruls.append(pred.item())
            cycle_points.append(cycles[i - 1])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(cycle_points, predicted_ruls, color="#2563eb", linewidth=2, label="Predicted RUL")
    ax.axhline(y=RUL_CAP, color="gray", linestyle="--", alpha=0.5, label=f"RUL cap ({RUL_CAP})")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Predicted Remaining Useful Life (cycles)")
    ax.set_title(f"Engine {selected_engine} — Live RUL Prediction")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    m1, m2, m3 = st.columns(3)
    m1.metric("Cycles Observed", f"{len(features)}")
    m2.metric("Latest Predicted RUL", f"{predicted_ruls[-1]:.1f} cycles")
    m3.metric("Peak Predicted RUL", f"{max(predicted_ruls):.1f} cycles")

st.divider()

# -------------------------
# Section: Manual Try-It-Yourself
# -------------------------

st.header("🎛️ Try It Yourself — Manual Sensor Input")
st.markdown(
    "Adjust sensor values (scaled 0-1, matching training data) to get a single RUL prediction. "
    "This treats your input as a steady snapshot repeated across the model's required 30-cycle "
    "window — a simplification made for manual demo input. The Engine Explorer above uses real "
    "historical sequences and better represents the model's actual use case."
)

default_values = test_df[feature_cols].mean().to_dict()

manual_cols = st.columns(3)
manual_inputs = {}
for idx, col in enumerate(feature_cols):
    with manual_cols[idx % 3]:
        manual_inputs[col] = st.slider(
            col, min_value=0.0, max_value=1.0, value=round(float(default_values[col]), 3), step=0.01
        )

if st.button("Predict RUL"):
    snapshot = np.array([manual_inputs[c] for c in feature_cols], dtype=np.float32).reshape(1, -1)
    window = np.repeat(snapshot, SEQUENCE_LENGTH, axis=0)
    window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred = model(window_tensor)
    st.success(f"Predicted RUL: **{pred.item():.1f} cycles remaining**")

st.divider()

# -------------------------
# Section: Model performance
# -------------------------

st.header("📊 Model Performance")
st.markdown("LSTM compared against a feedforward baseline that ignores sequence order:")

perf_df = pd.DataFrame({
    "Metric": ["Test RMSE (cycles)", "PHM Score (asymmetric, lower is better)"],
    "Baseline (Dense NN)": [14.39, 433.59],
    "LSTM": [13.23, 246.35],
})
st.table(perf_df)

st.markdown(
    "The LSTM's advantage is modest on RMSE (~8%) but substantial on the PHM score (~43%) — "
    "indicating it makes meaningfully fewer dangerously-late predictions, the operationally "
    "costly failure mode in real prognostics systems."
)

st.divider()
st.caption("Built with PyTorch (LSTM) · NASA C-MAPSS FD001 dataset · Streamlit")
