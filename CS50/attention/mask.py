import sys
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer, TFBertForMaskedLM

"""
$ python mask.py
$ check50 --local ai50/projects/2024/x/attention
"""
# Pre-trained masked language model
MODEL = "bert-base-uncased"

# Number of predictions to generate
K = 3

# Constants for generating attention diagrams
FONT = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 28)
GRID_SIZE = 40
PIXELS_PER_WORD = 200

# Hide GPU from visible devices
def InitializeGPU():
    """
    2024-12-17 12:39:33.030218: I external/local_xla/xla/stream_executor/cuda/cuda_driver.cc:1193] failed to allocate 2.2KiB (2304 bytes) from device: RESOURCE_EXHAUSTED: : CUDA_ERROR_OUT_OF_MEMORY: out of memory
    https://stackoverflow.com/questions/39465503/cuda-error-out-of-memory-in-tensorflow
    https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
    https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth
    https://www.tensorflow.org/guide/gpu
    """
    #tf.config.set_visible_devices([], 'GPU')
    #tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_physical_devices('GPU')
    print(f"{len(gpus)} GPUs available")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def main():
    text = input("Text: ")

    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    inputs = tokenizer(text, return_tensors="tf")
    mask_token_index = get_mask_token_index(tokenizer.mask_token_id, inputs)
    if mask_token_index is None:
        sys.exit(f"Error! Input must include mask token {tokenizer.mask_token}!")

    # Use model to process input
    model = TFBertForMaskedLM.from_pretrained(MODEL)
    result = model(**inputs, output_attentions=True)

    # Generate predictions
    mask_token_logits = result.logits[0, mask_token_index]
    top_tokens = tf.math.top_k(mask_token_logits, K).indices.numpy()
    for token in top_tokens:
        print(text.replace(tokenizer.mask_token, tokenizer.decode([token])))

    # Visualize attentions
    visualize_attentions(inputs.tokens(), result.attentions)

def get_mask_token_index(mask_token_id, inputs):
    """
    Return the index of the token with the specified `mask_token_id`, or
    `None` if not present in the `inputs`.

    inputs: {'input_ids': <tf.Tensor: shape=(1, 12), dtype=int32, numpy=array([[ 101, 2059, 1045, 3856, 2039, 1037,  103, 2013, 1996, 2795, 1012, 102]], dtype=int32)>, 
             'token_type_ids': <tf.Tensor: shape=(1, 12), dtype=int32, numpy=array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int32)>, 
             'attention_mask': <tf.Tensor: shape=(1, 12), dtype=int32, numpy=array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=int32)>}
    """
    #print(f"mask_token_id: {mask_token_id}, input ids: {inputs.input_ids}")
    for ids in inputs.input_ids:
        index = 0
        for id in ids:
            if id == mask_token_id:
                return index
            index += 1
    return None

def get_color_for_attention_score(attention_score):
    """
    Return a tuple of three integers representing a shade of gray for the
    given `attention_score`. Each value should be in the range [0, 255].
    """
    return (int(attention_score * 255),int(attention_score * 255),int(attention_score * 255))

def visualize_attentions(tokens, attentions):
    """
    Produce a graphical representation of self-attention scores.

    For each attention layer, one diagram should be generated for each
    attention head in the layer. Each diagram should include the list of
    `tokens` in the sentence. The filename for each diagram should
    include both the layer number (starting count from 1) and head number
    (starting count from 1).

    To index into the attentions value to get a specific attention head’s values, you can do so as attentions[i][j][k], where i is the index of the attention layer, j is the index of the beam number (always 0 in our case), and k is the index of the attention head in the layer.    
attentions: (<tf.Tensor: shape=(1, 12, 11, 11), dtype=float32, numpy=
array([[[[3.62220705e-02, 5.82316928e-02, 2.76106466e-02, ...,
          8.89380723e-02, 2.87756119e-02, 1.85958862e-01],
         [1.29470080e-01, 6.55574948e-02, 9.98986438e-02, ...,
          5.04852198e-02, 8.23077187e-02, 8.15437585e-02],
         [1.40034065e-01, 4.85403873e-02, 1.57218903e-01, ...,
          3.91530320e-02, 1.33768111e-01, 7.21059516e-02],
         ...,
         [6.40155673e-02, 6.72601163e-02, 1.01721503e-01, ...,
          1.10172629e-01, 5.50704226e-02, 6.41582012e-02],
         [2.92556342e-02, 7.52940550e-02, 1.43501610e-01, ...,
          6.31291494e-02, 1.09347329e-01, 1.20522656e-01],
         [8.30162391e-02, 4.55733351e-02, 7.41427764e-02, ...,
          1.00928277e-01, 6.89814761e-02, 1.28097951e-01]],
]]))
    """
    print(f"{len(attentions)} layers, {len(attentions[0])} beams, {len(attentions[0][0])} heads")
    #print(f"attentions: {attentions}")
    for l in range(len(attentions)):
        for b in range(len(attentions[0])):
            for h in range(len(attentions[l][b])):
                generate_diagram(
                    l+1,
                    h+1,
                    tokens,
                    attentions[l][b][h]
                )

def generate_diagram(layer_number, head_number, tokens, attention_weights):
    """
    Generate a diagram representing the self-attention scores for a single
    attention head. The diagram shows one row and column for each of the
    `tokens`, and cells are shaded based on `attention_weights`, with lighter
    cells corresponding to higher attention scores.

    The diagram is saved with a filename that includes both the `layer_number`
    and `head_number`.
    """
    # Create new image
    image_size = GRID_SIZE * len(tokens) + PIXELS_PER_WORD
    img = Image.new("RGBA", (image_size, image_size), "black")
    draw = ImageDraw.Draw(img)

    # Draw each token onto the image
    for i, token in enumerate(tokens):
        # Draw token columns
        token_image = Image.new("RGBA", (image_size, image_size), (0, 0, 0, 0))
        token_draw = ImageDraw.Draw(token_image)
        token_draw.text(
            (image_size - PIXELS_PER_WORD, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )
        token_image = token_image.rotate(90)
        img.paste(token_image, mask=token_image)

        # Draw token rows
        _, _, width, _ = draw.textbbox((0, 0), token, font=FONT)
        draw.text(
            (PIXELS_PER_WORD - width, PIXELS_PER_WORD + i * GRID_SIZE),
            token,
            fill="white",
            font=FONT
        )

    # Draw each word
    for i in range(len(tokens)):
        y = PIXELS_PER_WORD + i * GRID_SIZE
        for j in range(len(tokens)):
            x = PIXELS_PER_WORD + j * GRID_SIZE
            color = get_color_for_attention_score(attention_weights[i][j])
            draw.rectangle((x, y, x + GRID_SIZE, y + GRID_SIZE), fill=color)

    # Save image
    img.save(f"Attention_Layer{layer_number}_Head{head_number}.png")

"""
Only executes when this file is run as a script
Does NOT execute when this file is imported as a module
__name__ stores the name of a module when it is loaded. It is set to either the string of "__main__" if it's in the top-level or the module's name if it is being imported.
"""
if __name__ == "__main__":
    InitializeGPU()
    main()
