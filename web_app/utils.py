import cv2
import streamlit as st


def setup_parameters():
    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 350px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 350px;
        margin-left: -350px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("Demo facial landmark")
    st.sidebar.subheader("Parameters")

    model_name = st.sidebar.selectbox("Choose the model", ["BasicCNN", "Mediapipe"])

    st.title(f"Facial landmark with {model_name}")

    st.set_option("deprecation.showfileUploaderEncoding", False)

    st.sidebar.markdown("---")
    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -400px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    detection_confidence = st.sidebar.slider("Min Detection Confidence", min_value=0.0, max_value=1.0, value=0.8)
    st.sidebar.markdown("---")

    st.markdown(" ## Output")

    return model_name, detection_confidence


def setup_annotation(vid, tfflie):
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = int(vid.get(cv2.CAP_PROP_FPS))

    st.sidebar.text("Input Video")
    st.sidebar.video(tfflie.name)

    fps = 0

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown("**FrameRate**")
        kpi1_text = st.markdown("0")
    with kpi2:
        st.markdown("**Image Height**")
        kpi2_text = st.markdown("0")
    with kpi3:
        st.markdown("**Image Width**")
        kpi3_text = st.markdown("0")

    st.markdown("<hr/>", unsafe_allow_html=True)

    return fps, width, height, fps_input, kpi1_text, kpi2_text, kpi3_text
