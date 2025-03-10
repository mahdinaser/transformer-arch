import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Embedding, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D, TimeDistributed
import numpy as np
import matplotlib.pyplot as plt

class TransformerEncoderLayer(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential([
            Dense(feed_forward_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
    
    def call(self, inputs, training):
        q = inputs
        k = inputs
        v = inputs
        attn_output = self.mha(q, v, k)  # Use explicit q, k, v
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class CustomSentenceTransformer(keras.Model):
    def __init__(self, vocab_size=20000, embedding_dim=128, num_heads=8, feed_forward_dim=512, num_layers=2):
        super(CustomSentenceTransformer, self).__init__()
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)
        self.encoder_layers = [TransformerEncoderLayer(embedding_dim, num_heads, feed_forward_dim) for _ in range(num_layers)]
        self.pooling = GlobalAveragePooling1D()
    
    def call(self, inputs, training=False):
        x = self.embedding(inputs)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, training=training)
        embeddings = self.pooling(x)
        return embeddings, x  # Return both pooled and sequence outputs

class MultiTaskModel(keras.Model):
    def __init__(self, vocab_size=20000, embedding_dim=128, num_heads=8, feed_forward_dim=512, num_layers=2, num_classes_task_a=4, num_named_entities=5):
        super(MultiTaskModel, self).__init__()
        self.sentence_transformer = CustomSentenceTransformer(vocab_size, embedding_dim, num_heads, feed_forward_dim, num_layers)
        
        # Task A: Sentence Classification
        self.task_a_classifier = Dense(num_classes_task_a, activation='softmax')
        
        # Task B: Named Entity Recognition (NER)
        self.ner_decoder = TimeDistributed(Dense(num_named_entities, activation='softmax'))
    
    def call(self, inputs):
        pooled_embeddings, sequence_embeddings = self.sentence_transformer(inputs)
        task_a_output = self.task_a_classifier(pooled_embeddings)  # Classification from pooled embeddings
        task_b_output = self.ner_decoder(sequence_embeddings)  # NER from sequence embeddings
        return {"task_a": task_a_output, "task_b": task_b_output}  # Named outputs

# Dummy Data for Training
train_texts = [
    "Apple is a tech company.",
    "Barack Obama was the 44th president of the United States.",
    "The Eiffel Tower is in Paris.",
    "Google acquired YouTube in 2006."
]
train_labels = np.array([0, 1, 2, 3])  # 0: Tech, 1: Politics, 2: Landmarks, 3: Business Acquisitions

train_ner_labels = [
    [1, 0, 0, 2, 0],
    [1, 1, 0, 0, 0, 0, 0, 2, 2, 2],
    [3, 3, 0, 0, 2],
    [1, 0, 1, 0, 0]
]

# Tokenization
def tokenize_sentences(sentences, vocab_size=20000):
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
    return padded_sequences, tokenizer

inputs_cls, tokenizer = tokenize_sentences(train_texts)
train_ner_labels = keras.preprocessing.sequence.pad_sequences(train_ner_labels, padding='post')
train_cls_labels = keras.utils.to_categorical(train_labels, num_classes=4)

# Define and Compile Model
model = MultiTaskModel()
model.compile(optimizer="adam", loss={"task_a": "categorical_crossentropy", "task_b": "sparse_categorical_crossentropy"})

# Training with Early Stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
history = model.fit(inputs_cls, {"task_a": train_cls_labels, "task_b": train_ner_labels}, epochs=200, callbacks=[early_stopping])

# Plot Training Loss
plt.plot(history.history['loss'], label='Total Loss')
plt.plot(history.history['task_a_loss'], label='Task A Loss')
plt.plot(history.history['task_b_loss'], label='Task B Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss Over Epochs')
plt.show()

# Prediction Function
def predict(sentences):
    tokenized_inputs, _ = tokenize_sentences(sentences)
    predictions = model.predict(tokenized_inputs)
    task_a_preds = np.argmax(predictions["task_a"], axis=1)
    task_b_preds = np.argmax(predictions["task_b"], axis=-1)
    return task_a_preds, task_b_preds

# Example Prediction
sample_texts = ["Elon Musk founded SpaceX.", "Microsoft acquired LinkedIn."]
predictions = predict(sample_texts)
class_labels = {0: "Tech", 1: "Politics", 2: "Landmarks", 3: "Business Acquisitions"}
print("Sentence Classification Predictions:")
for i, text in enumerate(sample_texts):
    print(f'"{text}" -> {class_labels[predictions[0][i]]}')
ner_labels = {0: "O", 1: "Person", 2: "Location", 3: "Organization", 4: "Miscellaneous"}
print("NER Class Labels: 0 -> Outside, 1 -> Person, 2 -> Location, 3 -> Organization, 4 -> Miscellaneous")
print("NER Predictions:")
for i, text in enumerate(sample_texts):
    words = text.split()
    ner_predictions = predictions[1][i][:len(words)]
    ner_result = [(word, ner_labels[ner_predictions[j]]) for j, word in enumerate(words)]
    print(f'"{text}" -> {ner_result}')
