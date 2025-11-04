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
  16/16 ━━━━━━━━━━━━━━━━━━━━ 11s 366ms/step - accuracy: 0.9020 - loss: 0.7031
  ```
- Example predictions:
  ```
  1/1 ━━━━━━━━━━━━━━━━━━━━ 7s 7s/step
  Class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = [[-2.2687287   2.0871005   3.2084212  -1.1271989   2.0089376   1.8511143
    0.33222625  0.26400954  0.2513574   0.39643517]]
  Truth: 2, Class: 2
  1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 48ms/step
  Class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = [[-0.5481384  -0.85897994 -0.6675978  -0.05240295  0.77217174  0.6485491
    0.84364176 -0.7001463   1.0141901   0.0160929 ]]
  Truth: 1, Class: 8
  1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 58ms/step
  Class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = [[ 0.01400222 -4.0995603   2.0077043  -2.7749326   2.4645112   1.2298834
    -0.66307074  0.01086101  0.72078615  0.9924923 ]]
  Truth: 3, Class: 4
  1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 51ms/step
  Class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = [[-0.8573711  -3.8974252   1.0792496  -2.3828738   1.4692718   4.228506
    0.12272664 -0.36451092 -0.14466111  1.0894351 ]]
  Truth: 5, Class: 5
  1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 54ms/step
  Class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = [[-1.6869063  -2.911563    0.07253426 -1.0236456   1.3742349   4.4545403
    0.15644817  0.17935908 -0.43823817  0.17643405]]
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
