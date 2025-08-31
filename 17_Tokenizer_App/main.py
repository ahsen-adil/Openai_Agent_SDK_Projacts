import base64
import io
import json
from typing import List, Dict

import pandas as pd
import streamlit as st
import tiktoken

# -----------------------------
# Helpers
# -----------------------------
MODEL_TO_TIKTOKEN = {
    "GPT-4 (cl100k_base)": "gpt-4",
    "GPT-4 Turbo (cl100k_base)": "gpt-4-0125-preview",
    "GPT-4o family (o200k_base)": "gpt-4o",  # falls back to o200k_base
    "GPT-3.5 (cl100k_base)": "gpt-3.5-turbo",
    "Davinci-003 (p50k_base)": "text-davinci-003",
    "Codex (p50k_base)": "code-davinci-002",
    "GPT-2 (r50k_base)": "gpt2",
}

WHITESPACE_MAP = {
    " ": "·",      # middle dot for space
    "\n": "↵\n",   # show line break arrow + actual newline
    "\t": "⇥\t",   # tab marker + actual tab
}


def get_encoding(name_or_model: str) -> tiktoken.Encoding:
    """Return a tiktoken encoding for a given display name or model string."""
    # If it's a display name, map to a model; otherwise assume it's already a model
    model = MODEL_TO_TIKTOKEN.get(name_or_model, name_or_model)
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        # Fallback to cl100k_base for unknown models
        enc = tiktoken.get_encoding("cl100k_base")
    return enc


def visualize_token_bytes(b: bytes) -> str:
    """Make bytes readable; attempt utf-8 decode with fallbacks, and make whitespace visible."""
    s = b.decode("utf-8", errors="replace")
    for k, v in WHITESPACE_MAP.items():
        s = s.replace(k, v)
    return s


def tokenize_text(text: str, enc: tiktoken.Encoding) -> Dict:
    token_ids: List[int] = enc.encode(text)
    token_bytes: List[bytes] = [enc.decode_single_token_bytes(t) for t in token_ids]

    rows = []
    for idx, (tid, tb) in enumerate(zip(token_ids, token_bytes)):
        rows.append({
            "#": idx,
            "token_id": tid,
            "token_str": visualize_token_bytes(tb),
            "bytes_hex": tb.hex(),
            "byte_len": len(tb),
        })

    df = pd.DataFrame(rows)
    return {
        "count": len(token_ids),
        "ids": token_ids,
        "bytes_hex": [b.hex() for b in token_bytes],
        "table": df,
    }


def download_link(data: bytes, filename: str, label: str) -> str:
    b64 = base64.b64encode(data).decode()
    return f'**[⬇ {label}](data:file/octet-stream;base64,{b64}\" download=\"{filename}\")**'


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="GPT Tokenizer (tiktoken)", layout="wide")
st.title("GPT Tokenizer App")

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox(
        "Encoding (choose a model)",
        list(MODEL_TO_TIKTOKEN.keys()),
        index=0,
        help="GPT‑4 uses cl100k_base; GPT‑4o uses o200k_base"
    )
    enc = get_encoding(model_name)

    st.caption("Tip: Switch encodings to see how token splits change.")

st.subheader("Enter your prompt")
text = st.text_area(
    "Paste or type text:",
    height=220,
    placeholder="Type anything… (Urdu/English/emoji supported)",
)

if text:
    result = tokenize_text(text, enc)
    st.success(f"Total tokens: {result['count']}")

    # Show table
    st.dataframe(result["table"], use_container_width=True)

    # Show raw lists
    with st.expander("Show raw token IDs"):
        st.code(json.dumps(result["ids"], ensure_ascii=False, indent=2))

    with st.expander("Show token bytes (hex)"):
        st.code(json.dumps(result["bytes_hex"], ensure_ascii=False, indent=2))

    # Downloads
    export = {
        "model": model_name,
        "count": result["count"],
        "ids": result["ids"],
        "bytes_hex": result["bytes_hex"],
    }
    json_bytes = json.dumps(export, ensure_ascii=False, indent=2).encode("utf-8")
    csv_bytes = result["table"].to_csv(index=False).encode("utf-8")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download JSON", data=json_bytes, file_name="tokens.json")
    with c2:
        st.download_button("Download CSV", data=csv_bytes, file_name="tokens.csv")

else:
    st.info("Enter some text above to see tokenization details.")

st.caption("Built with Streamlit + tiktoken. No API calls are made.")
