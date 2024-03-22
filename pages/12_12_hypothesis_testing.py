import streamlit as st

from utils.other_utils import add_logo, setup_sticky_header, add_title

def main():
    # Set up the layout of Streamlit app
    st.set_page_config(
        page_title='Hypothesis Testing',
        page_icon=':bar_chart:',
        layout='wide',
        initial_sidebar_state='collapsed'
    )

    # Add USF logo at sidebar
    add_logo()

    # Define sticky header
    header = st.container()
    with header:
        title = 'Hypothesis Testing'
        add_title(title)

        st.write("<div class='fixed-header'/>", unsafe_allow_html=True)

    # Set up sticky header
    setup_sticky_header(header)


if __name__ == '__main__':
    main()