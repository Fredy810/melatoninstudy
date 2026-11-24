import base64
import hmac
import zlib

import numpy as np
import streamlit as st


# =============================================================================
# Stage 1: Configuration, Scaler Statistics, and Model Metadata
# =============================================================================

DEFAULT_PASSWORD = "EFT1234!"
MELATONIN_MW = 232.28  # g/mol, used for mg/L -> uM dose standardization
UNKNOWN_LABEL = "Not provided / Unknown"

CATEGORICAL_COLS = [
    "Species",
    "Culture_system",
    "Explant/source",
    "Co-treatment/context",
    "Metabolite_class",
    "Metabolite/endpoint",
    "Analytical_method",
    "Unit",
]

# Standardization statistics fitted strictly on the training split (205 rows)
NUM_MEANS = {
    "Dose_uM_standardized": 20.02301,
    "log_dose": 2.482826,
    "log_control": -0.316506,
}

NUM_STDS = {
    "Dose_uM_standardized": 22.993666,
    "log_dose": 1.139513,
    "log_control": 2.067426,
}

# One-hot vocabularies from training split (each column adds +1 slot for unknown/missing)
CAT_VOCAB = {
    "Species": [
        "Berberis vulgaris L.",
        "Lepidium sativum L.",
        "Ocimum basilicum L.",
        "Ocimum basilicum L. var. purpurascens",
        "Ocimum basilicum L. var. thyrsiflora",
        "Rosmarinus officinalis L.",
        "Silybum marianum (L.) Gaertn.",
    ],
    "Culture_system": [
        "Callus",
        "Cell suspension",
        "Leaf-derived callus",
    ],
    "Explant/source": [
        "Callus",
        "Leaf",
        "Suspension-cultured cells",
    ],
    "Co-treatment/context": [
        "Callus culture; MEL elicitation",
        "Continuous light (24 h)",
        "Continuous light; light × melatonin factorial",
        "Dark; light × melatonin factorial",
        "MS + 1.0 mg/L BAP + 2.0 mg/L NAA",
        "Melatonin + UV-C 20 min",
        "Melatonin alone; NAA 2.5 mg/L basal medium",
        "Melatonin-alone arm",
        "Melatonin-alone arm; TDZ+NAA basal callus medium",
        "Melatonin-alone treatment",
        "Photoperiod; light × melatonin factorial",
        "TDZ 9.0 µM; MEL vs TDZ-only control",
    ],
    "Metabolite_class": [
        "Alkaloid pool",
        "Anthocyanin",
        "Flavonoid",
        "Flavonoid pool",
        "Flavonoid precursor",
        "Flavonolignan",
        "Phenolic acid",
        "Phenolic acids",
        "Phenolic pool",
        "Polyphenol",
        "Polyphenol pool",
        "Silymarin pool",
    ],
    "Metabolite/endpoint": [
        "Apigenin",
        "Caffeic acid",
        "Catechin",
        "Chicoric acid",
        "Chlorogenic acid",
        "Cinnamic acid",
        "Cyanidin",
        "Ferulic acid",
        "Gallic acid",
        "Isosilybin A",
        "Isosilybin B",
        "Isosilychristin",
        "Kaempferol",
        "Myricetin",
        "Naringenin",
        "Peonidin",
        "Protocatechuic acid",
        "Quercetin",
        "Resveratrol",
        "Rosmarinic acid",
        "Salicylic acid",
        "Silybin A",
        "Silybin B",
        "Silychristin",
        "Silydianin",
        "Sinapic acid",
        "Taxifolin",
        "Total alkaloid",
        "Total flavonoid content",
        "Total phenolic content",
        "Total quantified phenolic acids",
        "Total quantified polyphenols",
        "Total silymarin content",
        "Vanillic acid",
        "p-Coumaric acid",
    ],
    "Analytical_method": [
        "Colorimetric",
        "Colorimetric/assay",
        "Folin-Ciocalteu",
        "HPLC",
        "HPLC-PDA",
    ],
    "Unit": [
        "mg GAE g^-1 DW",
        "mg g^-1 DW",
        "mg g^-1 FW",
        "µg g^-1",
        "µg g^-1 FW",
    ],
}

UI_OPTIONS = {
    **CAT_VOCAB,
    "Metabolite/endpoint": sorted(
        CAT_VOCAB["Metabolite/endpoint"] + ["Total phenolic production", "Vitexin"]
    ),
    "Unit": sorted(CAT_VOCAB["Unit"] + ["mg GAE L^-1"]),
}

ENDPOINT_ALIASES = {
    "total phenolics": "Total phenolic content",
    "total flavonoids": "Total flavonoid content",
    "total alkaloid content": "Total alkaloid",
}

TEST_METRICS = {
    "train_rows": 205,
    "val_rows": 37,
    "test_rows": 61,
    "usable_rows": 303,
    "study_count": 7,
    "species_count": 7,
    "log_treatment_r2": 0.9652,
    "lrr_r2_full": 0.5484,
    "lrr_rmse_full": 0.4411,
    "lrr_mae_full": 0.2648,
    "lrr_r2_partial": 0.5086,
    "lrr_r2_no_control": 0.5377,
}


def get_app_password() -> str:
    """Read password from Streamlit Cloud secrets if configured, otherwise use default."""
    try:
        return str(st.secrets["APP_PASSWORD"])
    except Exception:
        return DEFAULT_PASSWORD


# =============================================================================
# Stage 2: Embedded Pre-Trained Neural Network Weights (95 -> 32 -> 16 -> 1)
# =============================================================================

PARAM_LAYOUT = [
    ("fc1.weight", (32, 95)),
    ("fc1.bias", (32,)),
    ("ln1.weight", (32,)),
    ("ln1.bias", (32,)),
    ("fc2.weight", (16, 32)),
    ("fc2.bias", (16,)),
    ("ln2.weight", (16,)),
    ("ln2.bias", (16,)),
    ("fc3.weight", (1, 16)),
    ("fc3.bias", (1,)),
]

# Pre-trained PyTorch float16 weights (3,713 parameters) packed with zlib + base64
EMBEDDED_WEIGHTS_B64 = (
    "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"
    "qnBJ+F9QbszgPnn+D8ro9vE="
)


@st.cache_resource
def unpack_model_weights() -> dict[str, np.ndarray]:
    """Decode and reshape the pre-trained weights once per Streamlit session."""
    raw_bytes = zlib.decompress(base64.b64decode(EMBEDDED_WEIGHTS_B64))
    flat_weights = np.frombuffer(raw_bytes, dtype=np.float16).astype(np.float32)

    weights = {}
    offset = 0
    for name, shape in PARAM_LAYOUT:
        count = int(np.prod(shape))
        weights[name] = flat_weights[offset : offset + count].reshape(shape)
        offset += count
    return weights


# =============================================================================
# Stage 3: Dose Standardization and Missing-Aware Feature Encoding
# =============================================================================

def dose_to_micromolar(dose_value: float, dose_unit: str) -> float:
    """Convert mg/L melatonin dose into uM using molecular weight 232.28 g/mol."""
    if dose_unit == "mg/L":
        return (dose_value / MELATONIN_MW) * 1000.0
    return dose_value


def encode_features(user_input: dict) -> tuple[np.ndarray, dict]:
    """
    Encode complete or partial experimental inputs into the 95-dimensional vector.
    Missing or unseen categories activate their dedicated <UNKNOWN> channel.
    """
    dose_um = user_input.get("Dose_uM_standardized")
    control_val = user_input.get("Control_value")

    if dose_um is not None and dose_um > 0:
        dose_z = (dose_um - NUM_MEANS["Dose_uM_standardized"]) / NUM_STDS["Dose_uM_standardized"]
        log_dose_z = (float(np.log1p(dose_um)) - NUM_MEANS["log_dose"]) / NUM_STDS["log_dose"]
        dose_missing = 0.0
    else:
        dose_z, log_dose_z, dose_missing = 0.0, 0.0, 1.0

    if control_val is not None and control_val > 0:
        log_ctrl_z = (float(np.log(control_val)) - NUM_MEANS["log_control"]) / NUM_STDS["log_control"]
        ctrl_missing = 0.0
    else:
        log_ctrl_z, ctrl_missing = 0.0, 1.0

    parts = [
        np.array([dose_z, log_dose_z, dose_missing, log_ctrl_z, ctrl_missing], dtype=np.float32)
    ]

    recognized_fields = []
    unseen_or_missing_fields = []

    if dose_missing == 0.0:
        recognized_fields.append("Melatonin dose")
    else:
        unseen_or_missing_fields.append("Melatonin dose")

    if ctrl_missing == 0.0:
        recognized_fields.append("Control value")
    else:
        unseen_or_missing_fields.append("Control value")

    for col in CATEGORICAL_COLS:
        vocab = CAT_VOCAB[col]
        vec = np.zeros(len(vocab) + 1, dtype=np.float32)

        raw_text = str(user_input.get(col, "") or "").strip()
        if raw_text == UNKNOWN_LABEL:
            raw_text = ""

        if col == "Metabolite/endpoint" and raw_text:
            raw_text = ENDPOINT_ALIASES.get(raw_text.lower(), raw_text)

        lookup = {item.lower(): idx for idx, item in enumerate(vocab)}
        match_idx = lookup.get(raw_text.lower()) if raw_text else None

        if match_idx is not None:
            vec[match_idx] = 1.0
            recognized_fields.append(col)
        else:
            vec[len(vocab)] = 1.0
            unseen_or_missing_fields.append(col)

        parts.append(vec)

    coverage = {
        "recognized": recognized_fields,
        "unknown_or_missing": unseen_or_missing_fields,
    }
    return np.concatenate(parts, axis=0).reshape(1, -1), coverage


# =============================================================================
# Stage 4: Neural Network Forward Inference
# =============================================================================

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def predict_metabolite(user_input: dict, weights: dict[str, np.ndarray]) -> dict:
    """Compute log response ratio, fold change, % change, and absolute content (if control given)."""
    x, coverage = encode_features(user_input)

    h1 = x @ weights["fc1.weight"].T + weights["fc1.bias"]
    h1 = silu(layer_norm(h1, weights["ln1.weight"], weights["ln1.bias"]))

    h2 = h1 @ weights["fc2.weight"].T + weights["fc2.bias"]
    h2 = silu(layer_norm(h2, weights["ln2.weight"], weights["ln2.bias"]))

    pred_lrr = float((h2 @ weights["fc3.weight"].T + weights["fc3.bias"]).item())
    fold_change = float(np.exp(pred_lrr))
    pct_change = (fold_change - 1.0) * 100.0

    control_val = user_input.get("Control_value")
    predicted_treatment = (
        float(control_val) * fold_change
        if (control_val is not None and control_val > 0)
        else None
    )

    return {
        "predicted_treatment": predicted_treatment,
        "fold_change": fold_change,
        "pct_change": pct_change,
        "log_response_ratio": pred_lrr,
        "coverage": coverage,
    }


# =============================================================================
# Stage 5: Authentication and Streamlit User Interface
# =============================================================================

def check_password() -> bool:
    if st.session_state.get("authenticated", False):
        return True

    st.title("Login")
    st.caption("Enter the access password to use the Secondary Metabolite Predictor.")
    with st.form("login_form"):
        entered = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        expected = get_app_password()
        if hmac.compare_digest(entered, expected):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


def render_category_input(label: str, col_name: str, manual_mode: bool) -> str:
    if manual_mode:
        return st.text_input(
            label,
            value="",
            placeholder="Optional — leave blank or type custom value",
        )
    options = [UNKNOWN_LABEL] + UI_OPTIONS[col_name]
    return st.selectbox(label, options=options, index=0)


def main() -> None:
    st.set_page_config(
        page_title="Secondary Metabolite Predictor",
        layout="centered",
    )

    if not check_password():
        return

    weights = unpack_model_weights()

    st.title("Secondary Metabolite Content Predictor")
    st.caption(
        "Estimates in vitro plant culture secondary-metabolite response under exogenous "
        "melatonin treatment. Supports both complete and partial experimental inputs."
    )

    with st.sidebar:
        st.subheader("Hold-out Test Evaluation")
        st.write(f"ln(Treatment) R2: {TEST_METRICS['log_treatment_r2']:.4f}")
        st.write(f"Full inputs LRR R2: {TEST_METRICS['lrr_r2_full']:.4f}")
        st.write(f"35% missing inputs LRR R2: {TEST_METRICS['lrr_r2_partial']:.4f}")
        st.write(f"Without control LRR R2: {TEST_METRICS['lrr_r2_no_control']:.4f}")
        st.write(
            f"LRR RMSE: {TEST_METRICS['lrr_rmse_full']:.4f} | "
            f"MAE: {TEST_METRICS['lrr_mae_full']:.4f}"
        )

        st.divider()
        st.subheader("Training Summary")
        st.write(
            f"Split: {TEST_METRICS['train_rows']} train / "
            f"{TEST_METRICS['val_rows']} val / {TEST_METRICS['test_rows']} test"
        )
        st.write(
            f"Source data: {TEST_METRICS['usable_rows']} records "
            f"({TEST_METRICS['study_count']} studies, {TEST_METRICS['species_count']} species)"
        )
        st.write("Mode: Standalone (missing-aware neural network)")

        st.divider()
        if st.button("Sign out", use_container_width=True):
            st.session_state["authenticated"] = False
            st.rerun()

    manual_mode = st.checkbox(
        "Manual text entry mode (leave any box blank or type unseen species/metabolites)",
        value=False,
    )

    with st.form("prediction_form"):
        st.subheader("Experimental Parameters")
        st.caption(
            "Fill in as many fields as you know. Leave unknown parameters blank or set to "
            "'Not provided / Unknown'."
        )
        col1, col2 = st.columns(2)

        with col1:
            species = render_category_input("Species", "Species", manual_mode)
            culture_system = render_category_input("Culture System", "Culture_system", manual_mode)
            explant = render_category_input("Explant / Source", "Explant/source", manual_mode)
            co_treatment = render_category_input(
                "Co-treatment / Context", "Co-treatment/context", manual_mode
            )
            dose_unit = st.selectbox("Melatonin Dose Unit", ["uM", "mg/L"])
            dose_input = st.number_input(
                "Exogenous Melatonin Dose (optional)",
                min_value=0.0001,
                value=None,
                placeholder="e.g. 25.0 (leave blank if unknown)",
                format="%.4f",
            )

        with col2:
            metabolite_class = render_category_input(
                "Metabolite Class", "Metabolite_class", manual_mode
            )
            endpoint = render_category_input(
                "Metabolite / Endpoint", "Metabolite/endpoint", manual_mode
            )
            analytical_method = render_category_input(
                "Analytical Method", "Analytical_method", manual_mode
            )
            unit = render_category_input("Measurement Unit", "Unit", manual_mode)
            control_value = st.number_input(
                "Control Metabolite Content (optional baseline)",
                min_value=0.0001,
                value=None,
                placeholder="e.g. 0.58 (leave blank to predict %/fold only)",
                format="%.4f",
            )

        submitted = st.form_submit_button("Calculate", use_container_width=True)

    if submitted:
        dose_um = dose_to_micromolar(dose_input, dose_unit) if dose_input is not None else None
        user_input = {
            "Species": species,
            "Culture_system": culture_system,
            "Explant/source": explant,
            "Co-treatment/context": co_treatment,
            "Metabolite_class": metabolite_class,
            "Metabolite/endpoint": endpoint,
            "Analytical_method": analytical_method,
            "Unit": unit,
            "Dose_uM_standardized": dose_um,
            "Control_value": control_value,
        }

        result = predict_metabolite(user_input, weights)
        clean_unit = "" if unit in ("", UNKNOWN_LABEL) else unit.strip()
        display_unit = clean_unit if clean_unit else "control unit"

        st.divider()
        st.subheader("Prediction Result")

        if result["predicted_treatment"] is not None:
            m1, m2, m3 = st.columns(3)
            m1.metric(
                label=f"Predicted Content ({display_unit})",
                value=f"{result['predicted_treatment']:.4f}",
            )
            m2.metric(label="Fold Change", value=f"{result['fold_change']:.2f}x")
            m3.metric(label="Change vs. Control", value=f"{result['pct_change']:+.1f}%")
        else:
            m1, m2 = st.columns(2)
            m1.metric(label="Predicted Fold Change", value=f"{result['fold_change']:.2f}x")
            m2.metric(label="Predicted Change vs. Control", value=f"{result['pct_change']:+.1f}%")
            st.info(
                "Control baseline was not provided, so relative response (fold change and % change) "
                "is reported. Enter a control value to also compute absolute metabolite content."
            )

        dose_text = f"{dose_um:.4f} uM" if dose_um is not None else "Not provided"
        recognized_count = len(result["coverage"]["recognized"])
        total_fields = recognized_count + len(result["coverage"]["unknown_or_missing"])

        st.caption(
            f"Standardized melatonin dose: {dose_text} | "
            f"Internal ln(Treatment/Control): {result['log_response_ratio']:.4f} | "
            f"Matched inputs: {recognized_count}/{total_fields}"
        )


if __name__ == "__main__":
    main()
