import streamlit as st
import tempfile
from streamlit_image_comparison import image_comparison
import json 
import cv2
import time
import os
from streamlit_super_slider import st_slider

GT_COLOR   = (0,255,0)
KEPT_COLOR = (0,255,255)
REJ_COLOR  = (255,0,0)

# --- HELPER FUNCTIONS ---
def gt_norm_to_px(cx, cy, w, h, img_w=1280, img_h=720):
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2

def draw_boxes(frame, detections: list[dict], gt_boxes_norm: list[list[float]]) -> None:
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        color = KEPT_COLOR if det["kept"] else REJ_COLOR
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if det["kept"]:
            cv2.putText(
                frame,
                f"DRONE {det['fusion_score']:.2f}",
                (x1, max(y1 - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, KEPT_COLOR, 1, cv2.LINE_AA,
            )
    for box in gt_boxes_norm:
        cx, cy, bw, bh = box
        x1, y1, x2, y2 = gt_norm_to_px(cx, cy, bw, bh)
        cv2.rectangle(frame, (x1, y1), (x2, y2), GT_COLOR, 2)

def generate_video(jsonl_data, image_key, video_name, fps=30):
    """
    Creates a video from image paths stored in JSONL records.

    Args:
        jsonl_data (tuple | list): Sequence of dictionaries parsed from a JSONL file.
        image_key (str): Key containing the image path.
        video_name (str): Output video filename.
        fps (int | float): Frames per second.
    """
    # Extract image paths
    image_paths = [
        record[image_key]
        for record in jsonl_data
        if image_key in record and os.path.exists(record[image_key])
    ]

    if not image_paths:
        raise ValueError("No valid image paths found.")

    # Read first image to determine frame size
    frame = cv2.imread(image_paths[0])
    if frame is None:
        raise ValueError(f"Could not read image: {image_paths[0]}")

    height, width = frame.shape[:2]

    # Create video writer
    video = cv2.VideoWriter(
        video_name,
        cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
        fps,
        (width, height),
    )

    # Add images to the video
    for image_path in image_paths:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        # Resize if necessary
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))

        video.write(frame)

    video.release()
    cv2.destroyAllWindows()

def image_slider(img1_file, img2_file): # not really necessary now that we have image paths
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f1:
            f1.write(img1_file.getvalue())
            path1 = f1.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f2:
        f2.write(img2_file.getvalue())
        path2 = f2.name
    # path1=save_uploaded_file(img1_file)
    # path2=save_uploaded_file(img2_file)
    image_comparison(
        img1=path1,
        img2=path2,
        label1="RGB Frame",
        label2="Event Frame",
        width=800,
        starting_position=50,
        show_labels=True,
        make_responsive=True,
    )

st.set_page_config(page_title="Applied Machine Intelligence Project - Group 7", layout="wide")

st.title("Hybrid Vision - Group 7")
st.write("Web interface to show project results")
st.write("<- open sidebar to start")

# --- SIDEBAR ---
st.sidebar.title("Hybrid Vision - Group 7")
st.sidebar.header("Step 1 - Data Import")

uploaded_files = st.sidebar.file_uploader(
    "Upload dataset", accept_multiple_files="directory", type=["zip"]
)
if uploaded_files:
    st.session_state["dataset"] = uploaded_files
    # st.sidebar.success(f"{len(uploaded_files)} frames uploaded")

st.sidebar.header("Step 2 - Run Model")
st.sidebar.button("Run")

st.sidebar.header("Step 3 - Visualize!")
st.sidebar.info("Click a visualization tab at the top.")

# --- PROCESS DATASET ---
# if st.button("Run"):
#     # here run program, ignore for now
# for now: load input dataset
input_data=[]


# --- LOAD PROCESSED DATA ---
@st.cache_data
def load_jsonl(jsonl_path):
    with open(jsonl_path, "r") as f:
        return [json.loads(line) for line in f] 
processed_data = load_jsonl("outputs/web/fusion_detections_blind_test_v4.jsonl")
number_of_frames=len(processed_data)

# --- TOP MENU ---
tabs = st.tabs([
    "Dashboard", "Side-by-Side", "As a video", "confusion matrices", "Example"
])

# --- DASHBOARD TAB ---
with tabs[0]:
    st.subheader("Preview the uploaded dataset")

    col1,col2=st.columns(2)
    with col1:
        selected = st_slider(
            min_value=0,
            max_value=len(processed_data) - 1,
            key='dashboard'
        )
        # Let user pick which image to view
        current_frame=processed_data[selected]
        event_path=current_frame["event_image"]
        rgb_path=current_frame["rgb_image"]
        st.write('Number of uploaded frames:', number_of_frames)
        st.write('this is actually the processed data without the bounding boxes but it should be fine :)')
    with col2:
        image_comparison(
            img1=rgb_path,
            img2=event_path,
            label1="RGB Frame",
            label2="Event Frame",
            width=800,
            starting_position=50,
            show_labels=True,
            make_responsive=True,
        )


with tabs[1]:
    st.subheader("Here you can see the detections of the model for both event and rgb detections")
    selected = st_slider(
        min_value=0,
        max_value=len(processed_data) - 1,
        key='side'
    )

    current_frame=processed_data[selected]
    event_frame=cv2.imread(current_frame["event_image"])
    draw_boxes(event_frame,current_frame.get("detections", []), current_frame.get("gt_boxes_norm", []))
    rgb_frame=cv2.imread(current_frame["rgb_image"])
    draw_boxes(rgb_frame,current_frame.get("detections", []), current_frame.get("gt_boxes_norm", []))

    col1,col2=st.columns(2)
    with col1:
        st.write('event frames')
        st.image(event_frame)
    with col2:
        st.write('rgb frames')
        st.image(rgb_frame)
    st.write('here we write what the colors mean')

with tabs[3]:
    st.subheader("confusion matrices")
    st.image('outputs/confusion_matrices_blind_test_v4.png')

with tabs[4]:
    st.write('decide what data to keep in the repository as an example')