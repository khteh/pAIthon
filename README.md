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
  1/1 ━━━━━━━━━━━━━━━━━━━━ 6s 6s/step
  Class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = [[-2.0099866  -0.56881446  4.5948634  -2.6432185   2.2302372   1.7344184
    0.23657088  0.4649722   1.4646876  -0.11805859]]
  Truth: 2, Class: 2
  1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 42ms/step
  Class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = [[ 1.1785551  -1.3516254   0.6351655  -2.5380015  -1.3923934  -0.04097138
    0.8621799  -0.7493809   1.1925981  -0.79342335]]
  Truth: 1, Class: 8
  1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 41ms/step
  Class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = [[ 1.0378284  -3.540225   -1.0644982  -4.938359   -1.5945294   1.5830379
    -0.3573578  -2.0449076   1.9372692  -0.02878545]]
  Truth: 3, Class: 8
  1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 41ms/step
  Class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = [[ 0.39373627 -3.8968973  -0.22976398 -4.179752   -0.29858652  3.7758834
    0.9506325  -1.949536    1.0997412   0.07072492]]
  Truth: 5, Class: 5
  1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 45ms/step
  Class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = [[-1.0685589  -2.327231   -0.17754719 -3.8118033   3.1382742   3.5833392
    0.10239653 -0.71539956  1.3708843  -0.6684125 ]]
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

## Semantic Image Segmentation UNet

### Architecture:

![Image Segmentation UNet](images/ImageSegmentationUNetModel.png?raw=true "Image Segmentation UNet")

### Predictions:

![Image Segmentation Predictions](images/ImageSegmentationUNet.png?raw=true "Image Segmentation Predictions")

## Heart Disease Decision Tree Model

```
classes: [0 1]
Metrics train:
	Accuracy score: 0.8583
Metrics validation:
	Accuracy score: 0.8641
```

![Heart Disease Decision Tree Model](images/HeartDiseasePredictionDecisionTree.png?raw=true "Heart Disease Decision Tree Model")

## NHANES | epidemiology risk analysis

- Predicting the 10-year risk of death of individuals from the NHANES | epidemiology dataset.
- Model: Random Forest
- Dataset consists of `Age, Diastolic BP, Poverty index, Race, Red blood cells, Sedimentation rate, Serum Albumin, Serum Cholesterol, Serum Iron, Serum Magnesium, Serum Protein, Sex, Systolic BP, TIBC, TS, White blood cells, BMI, Pulse pressure`
- Model explaination of contributing factors to 10-year risk of death:
  - Red colors contribute positively and blue colors contribute negatively to the 10-year risk of death.
    ![NHANES / epidemiology risk analysis](images/NHANESEpidemiologyDeathFactors1.png?raw=true "NHANES / epidemiology risk analysis")
    ![NHANES / epidemiology risk analysis](images/NHANESEpidemiologyDeathFactors.png?raw=true "NHANES / epidemiology risk analysis")
    ![NHANES / epidemiology risk analysis](images/NHANESEpidemiologyDeathFactors_age_sex.png?raw=true "NHANES / epidemiology risk analysis")

## MNIST GAN

![MNIST GAN](images/mnist_gan.gif?raw=true "MNIST GAN")

## Sine Wave GAN

- https://medium.com/@khteh/sine-wave-generative-adversarial-network-gan-8858bf5c867a
  ![Sine Wave GAN](images/sinewave_gan.gif?raw=true "Sine Wave GAN")

## Harvard Extended CS50 course works

https://cs50.harvard.edu/ai/2024/
