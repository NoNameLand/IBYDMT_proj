import numpy as np
import open_clip
import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from streamlit_image_select import image_select

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

supported_models_path = hf_hub_download(
    repo_id="jacopoteneggi/IBYDMT",
    filename="supported_models.txt",
    repo_type="dataset",
)
SUPPORTED_MODELS = []
with open(supported_models_path, "r") as f:
    for line in f:
        line = line.strip()
        model_name, _ = line.split(",")
        SUPPORTED_MODELS.append(model_name)

st.set_page_config(
    layout="wide",
    initial_sidebar_state=st.session_state.get("sidebar_state", "collapsed"),
)
st.session_state.sidebar_state = "collapsed"
st.markdown(
    """
        <style>
            textarea {
                font-family: monospace !important;
            }
            input {
                font-family: monospace !important;
            }
        </style>
        """,
    unsafe_allow_html=True,
)

st.markdown(
    """
        # I Bet You Did Not Mean That

        Official HF Space for the paper [*I Bet You Did Not Mean That: Testing Semantci Importance via Betting*](https://arxiv.org/pdf/2405.19146)

        by [Jacopo Teneggi](https://jacopoteneggi.github.io) and [Jeremias Sulam](https://sites.google.com/view/jsulam).

        ---
        """,
)


def validate_class_name(class_name):
    if class_name is None:
        return (False, "Class name cannot be empty")
    if class_name.strip() == "":
        return (False, "Class name cannot be empty")
    return (True, None)


def validate_concepts(concepts):
    concepts = concepts.split("\n")
    concepts = [concept.strip() for concept in concepts]
    concepts = [concept for concept in concepts if concept != ""]
    if len(concepts) < 3:
        return (concepts, False, "You must provide at least 3 concepts")
    if len(concepts) > 10:
        return (concepts, False, "Maximum 10 concepts allowed")
    return (concepts, True, None)


def get_image():
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        image = uploaded_file or image_select(
            label="or select one",
            images=[
                "assets/demo/ace.jpg",
                "assets/demo/ace.jpg",
                "assets/demo/ace.jpg",
                "assets/demo/ace.jpg",
            ],
        )
    return Image.open(image)


def load_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    )
    tokenizer = open_clip.get_tokenizer("hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K")


def test(image, class_name, concepts, cardinality):
    pass


def main():
    columns = st.columns([0.40, 0.60])

    with columns[0]:
        st.selectbox("Choose a model to test", options=SUPPORTED_MODELS)

        row1 = st.columns(2)
        row2 = st.columns(2)

        with row1[0]:
            image = get_image()

            st.image(image, use_column_width=True)
        with row1[1]:
            class_name = st.text_input(
                "Class to test",
                help="Name of the class to build the zero-shot CLIP classifier with.",
                value="cat",
            )
            concepts = st.text_area(
                "Concepts to test (max 10)",
                help="List of concepts to test the predictions of the model with. Write one concept per line.",
                height=160,
                value="piano\ncute\nwhiskers\nmusic\nwild",
            )

            class_ready, class_error = validate_class_name(class_name)
            concepts, concepts_ready, concepts_error = validate_concepts(concepts)

            cardinality = st.slider(
                "Size of conditioning set",
                help="The number of concepts to condition model predictions on.",
                min_value=1,
                max_value=max(2, len(concepts) - 1),
                value=1,
                step=1,
                disabled=not concepts_ready,
            )

        with row2[0]:
            change_image_button = st.button("Change Image", use_container_width=True)
            if change_image_button:
                st.session_state.sidebar_state = "expanded"
                st.experimental_rerun()
        with row2[1]:
            ready = class_ready and concepts_ready

            error_message = ""
            if class_error is not None:
                error_message += f"- {class_error}\n"
            if concepts_error is not None:
                error_message += f"- {concepts_error}\n"

            test_button = st.button(
                "Test",
                help=None if ready else error_message,
                use_container_width=True,
                disabled=not ready,
            )


if __name__ == "__main__":
    main()
