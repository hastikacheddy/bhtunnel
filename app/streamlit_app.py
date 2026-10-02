"""bhtunnel interactive demo.

    streamlit run app/streamlit_app.py
"""

import streamlit as st

st.set_page_config(page_title="bhtunnel: black-hole tunnelling on qubits",
                   page_icon="🕳️", layout="wide")

pages = [
    st.Page("views/home.py", title="Overview", icon="🏠", default=True),
    st.Page("views/wavepacket.py", title="Wavepacket", icon="🌊"),
    st.Page("views/greybody.py", title="Greybody & Hawking", icon="🌡️"),
    st.Page("views/circuit.py", title="Circuit", icon="🔌"),
    st.Page("views/amplitude.py", title="Amplitude estimation", icon="📈"),
]
st.navigation(pages).run()
