import streamlit as st
from utils.other_utils import add_logo


def main():

    st.set_page_config(
        page_title='MSDS504',
        page_icon=':bar_chart:',
        layout='wide'
    )

    add_logo()

    st.title('MSDS504 Statistics & Probability')


if __name__ == '__main__':
    main()