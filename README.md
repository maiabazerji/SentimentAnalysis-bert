# Sentiment Analysis with BERT

This project implements sentiment analysis using BERT (Bidirectional Encoder Representations from Transformers). The dataset used is from Yelp reviews, and the analysis focuses on classifying reviews as positive or negative.

## Installation

To run this project, you'll need to install the necessary libraries. Use the following commands to install them:

```bash
pip install -q transformers[torch]
pip install -q sentencepiece
pip install -q datasets
pip install -q wordcloud
```

### Library Descriptions

- **Transformers**: Provides pre-trained transformer-based models, with support for PyTorch.
- **SentencePiece**: A tokenizer designed to handle the segmentation of text into subword units.
- **Datasets**: An easy-to-use interface for accessing and sharing large datasets.
- **WordCloud**: Used to generate word clouds, which are visual representations of text data.

## Dataset

The dataset used for this project is sourced from Yelp reviews. It can be loaded directly using the following code:

```python
import pandas as pd

# Load the dataset
url = "https://gitlab.com/valdanchev/data-storage-for-teaching-ml/-/raw/main/yelp_reviews_data_500.csv"
df = pd.read_csv(url)
```

The dataset consists of two columns: `text` and `label`, where the label is 0 for negative reviews and 1 for positive reviews.

## Data Preprocessing

1. **Remove Stopwords**:
   Common words such as "the", "is", and "and" are removed as they do not provide useful information for analysis.

2. **Stemming**:
   Words are reduced to their root form using the Porter Stemmer.

```python
from nltk.corpus import stopwords
import nltk
from nltk.stem import PorterStemmer

nltk.download('stopwords')
stop = stopwords.words('english')
stemmer = PorterStemmer()

# Preprocess the text
df['text_preprocessed'] = df['text'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split() if word not in stop]))
```

3. **Save Negative and Positive Reviews**:
   Negative and positive reviews are saved to separate CSV files for further analysis.

```python
df[df['label'] == 0].to_csv('yelp_reviews_data_500_negative.csv', index=False)
df[df['label'] == 1].to_csv('yelp_reviews_data_500_positive.csv', index=False)
```

4. **Word Cloud Visualization**:
   Word clouds are generated for both positive and negative reviews to visualize the most frequently occurring words.

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Plot positive words
positive_df = pd.read_csv('yelp_reviews_data_500_positive.csv')
wordcloud_positive = WordCloud(width=800, height=800, background_color='white', stopwords=stop, min_font_size=10).generate(' '.join(positive_df['text_preprocessed']))

plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud_positive)
plt.axis("off")
plt.show()
```

## Tokenization

The `transformers` library is used for tokenization. The BERT tokenizer is initialized, and the dataset is tokenized for training.

```python
from transformers import AutoTokenizer
import datasets

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Convert df to dataset
dataset_train = datasets.Dataset.from_pandas(df_train[['text', 'label']], split='train')
dataset_test = datasets.Dataset.from_pandas(df_test[['text', 'label']], split='test')

# Tokenize the dataset
dataset_train = dataset_train.map(tokenize_function, batched=True)
dataset_test = dataset_test.map(tokenize_function, batched=True)
```

## Model Training

The BERT model is initialized for sequence classification. Training arguments are defined, and the model is trained using the Trainer API from the `transformers` library.

```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize the model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./sentiment_analysis",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=5,
    weight_decay=0.01,
    evaluation_strategy='steps',
    logging_dir='./sentiment_analysis/logs',
    logging_steps=25,
    save_steps=25,
    eval_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_test,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()
```

## Model Evaluation

The model's performance is evaluated using accuracy, precision, recall, and F1-score metrics.

```python
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
```

## Conclusion

This project demonstrates the use of BERT for sentiment analysis on Yelp reviews. With preprocessing, tokenization, and model training, you can classify text data effectively.

## License

This project is licensed under the MIT License.
