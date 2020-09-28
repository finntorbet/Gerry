import tensorflow as tf
from bert import BertModelLayer
from bert.tokenization.bert_tokenization import FullTokenizer
import numpy as np

model = tf.keras.models.load_model("models/intentClassifier/intentClassifier.h5", custom_objects={"BertModelLayer": BertModelLayer})
tokenizer = FullTokenizer(vocab_file="models/intentClassifier/vocab.txt")
max_seq_len = 19
classes = ['complaint', 'changeOfEmail', 'changeOfAddress', 'other']

np.set_printoptions(suppress=True)

def classify_intent(text):
    sentences = [text]

    # Pre-process - Normalize the text
    pred_tokens = map(tokenizer.tokenize, sentences)
    pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
    pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))
    pred_token_ids = map(
        lambda tids: tids + [0] * (max_seq_len - len(tids)),
        pred_token_ids
    )
    pred_token_ids = np.array(list(pred_token_ids))

    # Make predictions
    predictions = model.predict(pred_token_ids)  # .argmax(axis=-1)

    # Log prediction values
    print(f"==========\n\t{text}")
    for pred in list(zip(classes, predictions.tolist()[0])):
        print(pred)

    # Return the prediciton
    return classes[predictions.argmax(axis=1)[0]]
    # return classes[predictions[0]]