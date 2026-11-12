import streamlit as st

from data import categorized_tickers

# --- CALLBACKs for widget synchronization ---
def sync_from_multiselect():
    """Updates the state of pills based on what is selected in the multiselect widget"""
    current_selected = set(st.session_state["selected_tickers"])
    for cat, tickers in categorized_tickers.items():
        st.session_state[f"pills_{cat}"] = [t for t in tickers if t in current_selected]

def sync_from_pills(cat, cat_tickers):
    """Updates the multiselect widget based on the selections made in the category pills"""
    pill_selected = set(st.session_state[f"pills_{cat}"])
    current_multi = set(st.session_state["selected_tickers"])
    
    # Remove all tickers belonging to this category from the overall list
    current_multi -= set(cat_tickers)
    # Add back the ones that are currently selected in the clicked category
    current_multi |= pill_selected
    
    st.session_state["selected_tickers"] = list(current_multi)