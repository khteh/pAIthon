# pAIthon

Python AI, ML, DL and NLP exploration playground.

## Run

- `pipenv run python -m <module>`
- Some keras (NN) models are saved and retrieved from `models/` folder.

## Model Diagnostics

### Tensorboard

- `pipenv run tensorboard --logdir <path>`

## Machine Translation

- Bidirection LSTM
- Translate from human-readable date locale string to ISO-format date string.
- Architecture:
  ![Machine Translation](images/MachineTranslation.png?raw=true "Machine Translation")
- Attention Map:
  ![Attention Map](images/MachineTranslationAttentionMap.png?raw=true "Attention Map")

## LSTM Name Entity Recognition

- Unidirection LSTM
- Identify name entities in input seqeunce
  ![LSTM Name Entity Recognition](images/LSTM_NameEntityRecognition.png?raw=true "LSTM Name Entity Recognition")
- Example:
  - Inpuut: "Peter Parker , the White House director of trade and manufacturing policy of U.S , said in an interview on Sunday morning that the White House was working to prepare for the possibility of a second wave of the coronavirus in the fall , though he said it wouldn ’t necessarily come"
  - Output:
    ```
    Peter: B-per
    Parker: I-per
    White: B-org
    House: I-org
    Sunday: B-tim
    morning: I-tim
    White: B-org
    House: I-org
    coronavirus: B-org
    ’t: I-per
    necessarily: I-per
    ```

## LSTM Emojifier

- Sentiment analysis of input sentence with an emoji output using GloVe vectors.
  ![LSTM Emojifier](images/LSTM_Emojifier.png?raw=true "LSTM Emojifier")
- Examples (Some of them are mislabelled):
  ```
  The meal was great!:  🍴
  I had a tough day!:  ❤️
  The job looks interesting!:  😄
  I had a great trip!:  🍴
  I learnt something new today!:  ❤️
  ```

## Siamese Neural Network

- Text similarity classifier
  ![Siamese NN](images/SiameseNN.png?raw=true "Siamese NN")
- Examples:
  1. "When will I see you?", "When can I see you again?"
     Prediction: True
  2. "Do they enjoy eating the desert?", "Do they like hiking in the desert?"
     Prediction: False

## Transformer Text Summarization

- Encoder-Decoder Transformer architecture

## Image Segmentation UNet

![Image Segmentation UNet](images/ImageSegmentationUNet.png?raw=true "Image Segmentation UNet")

## MNIST GAN

![MNIST GAN](images/mnist_gan.gif?raw=true "MNIST GAN")

## Sine Wave GAN

- https://medium.com/@khteh/sine-wave-generative-adversarial-network-gan-8858bf5c867a
  ![Sine Wave GAN](images/sinewave_gan.gif?raw=true "Sine Wave GAN")

## Harvard Extended CS50 course works

https://cs50.harvard.edu/ai/2024/
