import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

st.title("Image Generation with GAN Neural style transfer for fog environments")
st.write("By Suhith")


# Define the profile picture
profile_image = Image.open("264827932_747332529567316_8091726915573986136_n.jpg").resize((150, 150))


# Define the size of the circular image holder
# size = st.sidebar.slider("Image Size", min_value=50, max_value=500, value=200, step=10)

# # Define the border width and color of the circular image holder
# border_width = st.sidebar.slider("Border Width", min_value=0, max_value=50, value=10, step=1)
# border_color = st.sidebar.color_picker("Border Color", value="#ffffff")

# Define the background color of the circular image holder
# background_color = st.sidebar.color_picker("Background Color", value="#dddddd")

# Convert the image to a numpy array
image_array = np.array(profile_image)

# Resize the image to fit the circular image holder
image_resized = Image.fromarray(image_array).resize((size, size))

# Create a circular mask
mask = Image.new("L", (size, size), 0)
draw = ImageDraw.Draw(mask)
draw.ellipse((0, 0, 150, 150), fill=255)

# Apply the circular mask to the image
image_masked = ImageOps.fit(image_resized, mask.size, centering=(0.5, 0.5))
image_masked.putalpha(mask)

# Create a background image
background_image = Image.new("RGBA", (150, 150), "#dddddd")

# Add the border to the image
border_image = ImageOps.expand(image_masked, border=9, fill="#dddddd")

# Combine the background image and border image
final_image = Image.alpha_composite(background_image, border_image)

# Display the final circular image holder
st.image(final_image, caption="Developer Image", use_column_width=True)
# Define the sidebar content
# st.sidebar.image(profile_image, use_column_width=True)
st.sidebar.title("Suhith")
st.sidebar.write("Final Year Undergraduate")
st.sidebar.write("IIT (University of Westminster)")
# Load image stylization module.
@st.cache(allow_output_mutation=True)
def load_model():
  return hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

style_transfer_model = load_model()

def perform_style_transfer(content_image, style_image):
  # Convert to float32 numpy array, add batch dimension, and normalize to range [0, 1]
    content_image = tf.convert_to_tensor(content_image, np.float32)[tf.newaxis, ...] / 255.
    style_image = tf.convert_to_tensor(style_image, np.float32)[tf.newaxis, ...] / 255.
    
    output = style_transfer_model(content_image, style_image)
    stylized_image = output[0]
    
    return Image.fromarray(np.uint8(stylized_image[0] * 255))

# Upload content and style images.
content_image = st.file_uploader("Upload image of a Regular Environment")
style_image = st.file_uploader("Upload Image of a foggy environment")

# default images

st.header("Drag and Drop example images bellow if you dont have any images to upload")

cola, colb = st.columns(2)



image = Image.open("download.jpg").resize((150, 150))
image2 = Image.open("download (2).jpg").resize((150, 150))

# Display the image and enable download
cola.image(image, caption="Mangrove Image", use_column_width=False)
cola.download_button(label="Download Mangrove", data="download.jpg")

colb.image(image2, caption="Foggy Image", use_column_width=False)
colb.download_button(label="Download foggy", data="download (2).jpg")

# col1, col2, col3,col4 = st.columns(4)

# if col1.button("Couple on bench"):
#   content_image = "examples/couple_on_bench.jpeg"
#   style_image = "examples/starry_night.jpeg"

# if col2.button("Couple Walking"):
#   content_image = "examples/couple_walking.jpeg"
#   style_image = "examples/couple_watercolor.jpeg"

# if col3.button("Golden Gate Bridge"):
#   content_image = "examples/golden_gate_bridge.jpeg"
#   style_image = "examples/couple_watercolor.jpeg"

# if col4.button("Joshua Tree"):
#   content_image = "examples/joshua_tree.jpeg"
#   style_image = "examples/starry_night.jpeg"



if style_image and content_image is not None:
  col1, col2 = st.columns(2)

  content_image = Image.open(content_image)
  # It is recommended that the style image is about 256 pixels (this size was used when training the style transfer network).
  style_image = Image.open(style_image).resize((256, 256))

  col1.header("Mangrove Environment")
  col1.image(content_image, use_column_width=True)
  col2.header("Foggy Environment")
  col2.image(style_image, use_column_width=True)

  output_image=perform_style_transfer(content_image, style_image)

  st.header("Generated Output image: Style transfer")
  st.image(output_image, use_column_width=True)

# # scroll down to see the references
st.markdown("**Colab Notebook Links and References**")

st.markdown("<a href='https://colab.research.google.com/drive/1ixgyBAmz8984B4N7YW1xVWuT1aESArlW?usp=sharing' target='_blank'>Source code for Neural Network</a>", unsafe_allow_html=True)

# st.markdown("<a href='https://www.tensorflow.org/hub/tutorials/tf2_arbitrary_image_stylization' target='_blank'>2. Tutorial to implement Fast Neural Style Transfer using the pretrained model from TensorFlow Hub</a>  \n", unsafe_allow_html=True)

# st.markdown("<a href='https://huggingface.co/spaces/luca-martial/neural-style-transfer' target='_blank'>3. The idea to build a neural style transfer application was inspired from this Hugging Face Space </a>", unsafe_allow_html=True)
