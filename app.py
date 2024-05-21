import streamlit as st
import time
from bin.app_utils import generate_lyrics


def main():

    st.set_page_config(page_title="Lyrics GPT", layout="centered")
    st.title("GPT for Hindi Song Lyrics Generation")

    start2 = ":green[***Please enter the starting lines to generate the song lyrics***]"
    length2 = ":green[***Enter the size of lyrics to generate***]"

    with st.form("Submit text for analysis:"):
        starting_lyrics = st.text_area(start2,                              
            placeholder="Teri baato mein aisa uljha",
            max_chars=128)
    
        pred_lyrics_length = st.number_input(length2, value= 200, step = 10, placeholder="default 2000")
        submit_button = st.form_submit_button("Generate Lyrics", type="primary")

    if starting_lyrics =="":
        starting_lyrics ="Teri baato mein aisa uljha"

    def stream_data():
        for word in pred_lyrics.split(" "):
            yield word + " "
            time.sleep(0.05)

    if submit_button:
        with st.spinner(text='In progress'):
            pred_lyrics= generate_lyrics(context= starting_lyrics, output_len=pred_lyrics_length)
            st.success('Done')

        pred_lyrics  = pred_lyrics.replace("\n", "\n\n")

        st.header('Genreated Lyrics')
        with st.container(border =True):
            st.write_stream(stream_data)

if __name__ == "__main__":
    main()