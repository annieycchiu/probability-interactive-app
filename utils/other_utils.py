import streamlit as st

def add_logo():
    st.markdown(
        f"""
        <style>
            [data-testid="stSidebarNav"] {{
                background-image: url('data:image/png;base64,{encode_image_to_base64("./assets/USF_MSDS_logo.png")}');
                background-repeat: no-repeat;
                background-size: 60%;
                padding-top: 160px;
                background-position: 55px 30px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def encode_image_to_base64(image_path):
    import base64
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
    

def setup_sticky_header(header):
    header
  
    ### Custom CSS for the sticky header
    st.markdown(
        """
    <style>
        div[data-testid="stVerticalBlock"] div:has(div.fixed-header) {
            position: sticky;
            top: 2.875rem;
            background-color: white;
            z-index: 999;
        }
        .fixed-header {
            border-bottom: 1px solid black;
        }
    </style>
        """,
        unsafe_allow_html=True
    )


def add_title(title, notation):
    st.write(
        f"<span style='font-size:35px; font-weight:bold;'>", title, "</span>", 
        f"<span style='font-size:23px; font-weight:bold; margin-left: 20px;'>", notation, "</span>", 
        unsafe_allow_html=True)
    
    st.write('')


def add_exp_var(expectation_formula, variance_formula):
    _, col = st.columns([0.03, 0.97])
    with col:
        st.write(
            "<span style='font-size:18px; font-weight:bold;'>Expectation:</span>", 
            f"<span style='font-size:16px; font-weight:bold; margin-left: 10px;'>", expectation_formula, "</span>", 
            unsafe_allow_html=True)
        
        st.write(
            "<span style='font-size:18px; font-weight:bold;'>Variance:</span>", 
            f"<span style='font-size:16px; font-weight:bold; margin-left: 37px;'>", variance_formula, "</span>", 
            unsafe_allow_html=True)
        
        st.write(
                "<span style='font-size:18px; font-weight:bold;'>Parameters:</span>", 
                unsafe_allow_html=True)
        
    st.write('')


    