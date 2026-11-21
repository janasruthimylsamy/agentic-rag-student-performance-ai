
import streamlit as st

st.set_page_config(
    page_title="Student Performance AI",
    page_icon="🎓",
    layout="wide"
)

# Application title
st.title("🎓 Student Performance AI")
st.caption("Agentic RAG-powered Student Analytics Assistant")

# Sidebar navigation
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a page",
    [
        "Dashboard",
        "AI Assistant",
        "Student Analytics",
        "Alerts",
        "Policy Knowledge"
    ]
)

if page == "Dashboard":
    st.header("Dashboard")
    st.write("Welcome to your Student Performance AI platform.")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", "—")
    col2.metric("Courses", "—")
    col3.metric("AI Assistant", "Ready to configure")

    st.info(
        "Your student analytics dashboard will appear here."
    )

elif page == "AI Assistant":
    st.header("AI Assistant")
    question = st.text_input(
        "Ask a question about student performance"
    )

    if st.button("Ask AI"):
        if question:
            st.info("AI integration will be connected next.")
        else:
            st.warning("Please enter a question.")

elif page == "Student Analytics":
    st.header("Student Analytics")
    st.info("Student performance analysis will appear here.")

elif page == "Alerts":
    st.header("Student Alerts")
    st.info("Student alert monitoring will appear here.")

elif page == "Policy Knowledge":
    st.header("Policy Knowledge")
    st.info("Policy document retrieval will appear here.")
