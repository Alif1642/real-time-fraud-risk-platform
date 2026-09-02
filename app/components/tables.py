"""Table helpers."""
import pandas as pd
import streamlit as st


def show_recent(df: pd.DataFrame, n: int = 20) -> None:
    st.dataframe(df.tail(n), use_container_width=True, hide_index=True)
