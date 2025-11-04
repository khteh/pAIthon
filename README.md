# pAIthon

Python AI, ML, DL, NLP and GAN exploration playground.

## Run

- `pipenv run python -m <module>`
- Some keras models are saved and loaded from `models/` folder.

## Model Diagnostics

### Tensorboard

- `pipenv run tensorboard --logdir <path>`

## Signs Language Digits Multiclass Classification

- Use `ResNet152V2` pretrained model.
- Test dataset accuracy:
  ```
  16/16 ━━━━━━━━━━━━━━━━━━━━ 9s 308ms/step - accuracy: 0.8898 - loss: 0.4807
  ```
- Example predictions:
  ```
  Class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = [[-1.7949121  -0.9373978   5.988724   -1.3011266  -0.30785906  2.200963
    0.3628277   0.494485    0.6216742  -0.4496336 ]]
  Truth: 2, Class: 2
  1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step
  Class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = [[ 3.2523103 -4.3629775  3.9744818 -2.2258158 -3.5589416  2.2114348
    -1.5373676 -0.8665462 -1.4555418 -2.1932414]]
  Truth: 1, Class: 2
  1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 40ms/step
  Class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = [[ 1.6622998 -5.4129186  0.5597717 -2.9309149 -4.046567   3.205339
    -3.1271842 -1.589747  -1.0898973 -4.412473 ]]
  Truth: 3, Class: 5
  1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 39ms/step
  Class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = [[ 2.0677593 -4.4180837 -0.5090899 -3.1197011 -2.7705925  4.9895453
    -1.4480599 -2.0641685 -2.213564  -3.062584 ]]
  Truth: 5, Class: 5
  1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 48ms/step
  Class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = [[ 0.9651322 -7.367579   2.380285  -4.6137404 -3.5754316  8.407264
    -1.0734806 -3.2308764 -2.5940378 -5.1837626]]
  Truth: 5, Class: 5
  ```

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
