import tensorflow as tf, matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import DistilBertTokenizerFast, TFDistilBertForQuestionAnswering
from utils.GPU import InitializeGPU, SetMemoryLimit
class TransformerExtractiveQA():
    """
    The goal of *extractive* QA is to identify the portion of the text that contains the answer to a question. 
    For example, when tasked with answering the question 'When will Jane go to Africa?' given the text data 'Jane visits Africa in September', the question answering model will highlight 'September'.
    """
    _path:str = None
    _data = None
    _data_flattened = None
    _data_processed = None
    _qa_dataset = None
    _train_ds = None
    _test_ds = None
    _train_tfdataset = None
    _batch_size:int = None
    _epochs:int = None
    _model: TFDistilBertForQuestionAnswering = None
    def __init__(self, path:str, batchsize:int, epochs:int):
        self._path = path
        self._batch_size = batchsize
        self._epochs = epochs
        """
        Before feeding the texts to a Transformer model, you will need to tokenize your input using a 🤗 Transformer tokenizer.
        Tokenizer must match the Transformer model type!
        Use the 🤗 DistilBERT fast tokenizer, which standardizes the length of your sequence to 512 and pads with zeros.
        This matches the maximum length used when creating tags.
        """
        self._tokenizer = DistilBertTokenizerFast.from_pretrained('tokenizer/')
        self._PrepareData()

    def BuildTrainModel(self):
        self._model = TFDistilBertForQuestionAnswering.from_pretrained("models/QuestionAnswer", return_dict=False)
        loss_fn1 = tf.keras.losses.SparseCategoricalCrossentropy( from_logits=True)
        loss_fn2 = tf.keras.losses.SparseCategoricalCrossentropy( from_logits=True)
        opt = tf.keras.optimizers.Adam(learning_rate=3e-5)
        losses = []
        for epoch in range(self._epochs):
            print("Starting epoch: %d"% epoch )
            for step, (x_batch_train, y_batch_train) in enumerate(self._train_tfdataset):
                with tf.GradientTape() as tape:
                    answer_start_scores, answer_end_scores = self._model(x_batch_train)
                    loss_start = loss_fn1(y_batch_train['start_positions'], answer_start_scores)
                    loss_end = loss_fn2(y_batch_train['end_positions'], answer_end_scores)
                    loss = 0.5 * (loss_start + loss_end)
                losses.append(loss)
                grads = tape.gradient(loss, self._model.trainable_weights)
                opt.apply_gradients(zip(grads, self._model.trainable_weights))
                if step % 20 == 0:
                    print("Training loss (for one batch) at step %d: %.4f"% (step, float(loss_start)))
        plt.plot(losses)
        plt.show()
    
    def Predict(self, text:str, question:str):
        input_dict = self._tokenizer(text, question, return_tensors='tf')
        outputs = self._model(input_dict)
        start_logits = outputs[0]
        end_logits = outputs[1]
        all_tokens = self._tokenizer.convert_ids_to_tokens(input_dict["input_ids"].numpy()[0])
        answer = ' '.join(all_tokens[tf.math.argmax(start_logits, 1)[0] : tf.math.argmax(end_logits, 1)[0]+1])
        print(question, answer.capitalize())

    def _PrepareData(self):
        # Load a dataset and print the first example in the training set
        self._data = load_from_disk(self._path)
        print(self._data['train'][0])
        # To make the data easier to work with, you will flatten the dataset to transform it from a dictionary structure to a table structure.
        self._data_flattened = self._data.flatten()
        self._data_processed = self._data_flattened.map(self._get_question_and_facts)
        self._data_processed = self._data_processed.map(self._get_start_end_idx)
        self._qa_dataset = self._data_processed.map(self._tokenize_align)
        self._qa_dataset = self._qa_dataset.remove_columns(['story.answer', 'story.id', 'story.supporting_ids', 'story.text', 'story.type'])
        self._train_ds = self._qa_dataset['train']
        self._test_ds = self._qa_dataset['test']

        columns_to_return = ['input_ids','attention_mask', 'start_positions', 'end_positions']
        self._train_ds.set_format(type='tf', columns=columns_to_return)
        print(f"_train_ds: {self._train_ds.shape}, self._train_ds['start_positions']: {type(self._train_ds['start_positions'])}")
        print(self._train_ds[200])
        # https://github.com/huggingface/datasets/issues/7772
        # train_features = {x: self._train_ds[x] for x in ['input_ids', 'attention_mask']}
        # start_pos = tf.convert_to_tensor(self._train_ds['start_positions'], dtype=tf.int64)
        # start_pos = tf.reshape(start_pos, [-1, 1])
        # end_pos = tf.convert_to_tensor(self._train_ds['end_positions'], dtype=tf.int64)
        # end_pos = tf.reshape(start_pos, [-1, 1])
        # train_labels = {"start_positions": start_pos, # XXX: https://huggingface.co/docs/datasets/v1.3.0/torch_tensorflow.html
        #                'end_positions': end_pos} # XXX: https://huggingface.co/docs/datasets/v1.3.0/torch_tensorflow.html
        # self._train_tfdataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels)).batch(self._batch_size)
        self._train_tfdataset = self._train_ds.to_tf_dataset(
            columns=['input_ids','attention_mask'],
            label_cols=['start_positions','end_positions'],
            shuffle=True,
            batch_size=self._batch_size
        )                

    def _get_question_and_facts(self, story):
        dic = {}
        dic['question'] = story['story.text'][2]
        dic['sentences'] = ' '.join([story['story.text'][0], story['story.text'][1]])
        dic['answer'] = story['story.answer'][2]
        return dic
    
    def _get_start_end_idx(self, story):
        str_idx = story['sentences'].find(story['answer'])
        end_idx = str_idx + len(story['answer'])
        return {'str_idx':str_idx,
            'end_idx': end_idx}

    def _tokenize_align(self, example):
        encoding = self._tokenizer(example['sentences'], example['question'], truncation=True, padding=True, max_length=self._tokenizer.model_max_length)
        start_positions = encoding.char_to_token(example['str_idx'])
        end_positions = encoding.char_to_token(example['end_idx']-1)
        if start_positions is None:
            start_positions = self._tokenizer.model_max_length
        if end_positions is None:
            end_positions = self._tokenizer.model_max_length
        return {'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'start_positions': start_positions,
            'end_positions': end_positions}    
    
if __name__ == "__main__":
    InitializeGPU()
    SetMemoryLimit(4096)
    transformer = TransformerExtractiveQA("data/QuestionAnswer", 8, 3)
    transformer.BuildTrainModel()
    transformer.Predict('The hallway is south of the garden. The garden is south of the bedroom.', 'What is south of the bedroom?')
    transformer.Predict('The arctic is north of Singapore. Singapore is north of the antarctic. Japan is east of Singapore. Hawaii is west of Singapore', 'What is in the middle?')