import numpy as np
import gradio as gr
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from huggingface_hub import from_pretrained_keras


model = from_pretrained_keras("keras-io/low-light-image-enhancement", compile=False)
examples = ['got2.png', 'gotj.png', 'goti.png' ]

def get_enhanced_image(data, output):
    r1 = output[:, :, :, :3]
    r2 = output[:, :, :, 3:6]
    r3 = output[:, :, :, 6:9]
    r4 = output[:, :, :, 9:12]
    r5 = output[:, :, :, 12:15]
    r6 = output[:, :, :, 15:18]
    r7 = output[:, :, :, 18:21]
    r8 = output[:, :, :, 21:24]
    x = data + r1 * (tf.square(data) - data)
    x = x + r2 * (tf.square(x) - x)
    x = x + r3 * (tf.square(x) - x)
    enhanced_image = x + r4 * (tf.square(x) - x)
    x = enhanced_image + r5 * (tf.square(enhanced_image) - enhanced_image)
    x = x + r6 * (tf.square(x) - x)
    x = x + r7 * (tf.square(x) - x)
    enhanced_image = x + r8 * (tf.square(x) - x)
    return enhanced_image
    
    
def infer(original_image):
    image = keras.preprocessing.image.img_to_array(original_image)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    output = model.predict(image)
    output = get_enhanced_image(image, output)
    output_image = tf.cast((output[0, :, :, :] * 255), dtype=np.uint8)    
    output_image = Image.fromarray(output_image.numpy())
    return output_image
    

iface = gr.Interface(
    fn=infer,
    title="Zero-DCE for low-light image enhancement",
    description = "Implementing Zero-Reference Deep Curve Estimation for low-light image enhancement.",
    inputs=[gr.inputs.Image(label="Original Image", type="pil")],
    outputs=[gr.outputs.Image(label="Enhanced Image", type="numpy")],
    examples=examples,
    article = "**Original Author**: [Soumik Rakshit](https://github.com/soumik12345) <br>**HF Contribution**: [Harveen Singh Chadha](https://github.com/harveenchadha)<br>",
    ).launch(debug=True, enable_queue=False, cache_examples=True)
