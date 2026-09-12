# Developing a Deep Learning Model for NER using LSTM

## Aim

To develop a deep learning model using a Bidirectional LSTM (BiLSTM) for Named Entity Recognition (NER) using PyTorch.

## Algorithm

1. Import the required Python libraries.
2. Load the NER dataset from `ner_dataset.csv`.
3. Fill missing values using forward filling.
4. Extract the unique words and tags from the dataset.
5. Create mappings between words and their numerical indexes and tags and their indexes.
6. Group the words and tags according to their sentences.
7. Convert the words and tags into numerical sequences.
8. Pad the sequences to a maximum length of 50.
9. Split the data into training and testing sets using an 80:20 ratio.
10. Create a custom `NERDataset` and DataLoader.
11. Create a BiLSTM model with an embedding layer, bidirectional LSTM layer, and linear layer.
12. Use Cross Entropy Loss and Adam optimizer for training.
13. Train the model for 3 epochs.
14. Evaluate the model using a classification report.
15. Plot the training and validation loss.
16. Perform prediction on a test sentence and compare the true and predicted NER tags.

## Program

```python
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.nn.utils.rnn import pad_sequence
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and prepare data
data = pd.read_csv("ner_dataset.csv", encoding="latin1").ffill()

words = list(data["Word"].unique())
tags = list(data["Tag"].unique())

if "ENDPAD" not in words:
    words.append("ENDPAD")

word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}
idx2tag = {i: t for t, i in tag2idx.items()}

# Group words by sentences
class SentenceGetter:
    def __init__(self, data):
        self.grouped = data.groupby(
            "Sentence #", group_keys=False
        ).apply(
            lambda s: [(w, t) for w, t in zip(s["Word"], s["Tag"])]
        )
        self.sentences = list(self.grouped)

getter = SentenceGetter(data)
sentences = getter.sentences

# Encode sentences
X = [[word2idx[w] for w, t in s] for s in sentences]
y = [[tag2idx[t] for w, t in s] for s in sentences]

# Pad sequences
max_len = 50

X_pad = pad_sequence(
    [torch.tensor(seq) for seq in X],
    batch_first=True,
    padding_value=word2idx["ENDPAD"]
)

y_pad = pad_sequence(
    [torch.tensor(seq) for seq in y],
    batch_first=True,
    padding_value=tag2idx["O"]
)

X_pad = X_pad[:, :max_len]
y_pad = y_pad[:, :max_len]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_pad, y_pad, test_size=0.2, random_state=1
)

# Dataset class
class NERDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "input_ids": self.X[idx],
            "labels": self.y[idx]
        }

train_loader = DataLoader(
    NERDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    NERDataset(X_test, y_test),
    batch_size=32
)

# BiLSTM model
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size):
        super(BiLSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(
            vocab_size, embedding_dim
        )

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            bidirectional=True,
            batch_first=True
        )

        self.hidden2tag = nn.Linear(
            hidden_dim * 2,
            tagset_size
        )

    def forward(self, input_ids):
        embeds = self.word_embeddings(input_ids)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)

        return tag_space

# Model parameters
EMBEDDING_DIM = 64
HIDDEN_DIM = 64

model = BiLSTMTagger(
    len(word2idx) + 1,
    EMBEDDING_DIM,
    HIDDEN_DIM,
    len(tag2idx)
).to(device)

loss_fn = nn.CrossEntropyLoss(
    ignore_index=tag2idx["O"]
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.005
)

# Training function
def train_model(
    model,
    train_loader,
    test_loader,
    loss_fn,
    optimizer,
    epochs=3
):
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids)

            loss = loss_fn(
                outputs.view(-1, outputs.shape[-1]),
                labels.view(-1)
            )

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids)

                loss = loss_fn(
                    outputs.view(-1, outputs.shape[-1]),
                    labels.view(-1)
                )

                total_val_loss += loss.item()

        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(test_loader)

        train_losses.append(avg_train)
        val_losses.append(avg_val)

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss={avg_train:.4f}, "
            f"Val Loss={avg_val:.4f}"
        )

    return train_losses, val_losses

# Evaluation function
def evaluate_model(model, test_loader, X_test, y_test):
    model.eval()

    true_tags, pred_tags = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids)
            preds = torch.argmax(outputs, dim=-1)

            for i in range(len(labels)):
                for j in range(len(labels[i])):
                    if labels[i][j].item() != tag2idx["O"]:
                        true_tags.append(
                            idx2tag[labels[i][j].item()]
                        )

                        pred_tags.append(
                            idx2tag[preds[i][j].item()]
                        )

    print(classification_report(true_tags, pred_tags))

# Run training and evaluation
train_losses, val_losses = train_model(
    model,
    train_loader,
    test_loader,
    loss_fn,
    optimizer,
    epochs=3
)

evaluate_model(
    model,
    test_loader,
    X_test,
    y_test
)

# Plot loss
print("NER Training Results")

history_df = pd.DataFrame({
    "loss": train_losses,
    "val_loss": val_losses
})

history_df.plot(title="Loss Over Epochs")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Inference and prediction
i = 125

model.eval()

with torch.no_grad():
    sample = X_test[i].unsqueeze(0).to(device)
    output = model(sample)

    preds = torch.argmax(
        output,
        dim=-1
    ).squeeze().cpu().numpy()

true = y_test[i].numpy()

print(
    '{:<15} {:<10} {}\n{}'.format(
        'Word',
        'True',
        'Pred',
        '-' * 40
    )
)

for w_id, true_tag, pred_tag in zip(
    X_test[i],
    y_test[i],
    preds
):
    if w_id.item() != word2idx["ENDPAD"]:
        word = words[w_id.item() - 1]

        true_label = tags[true_tag.item()]
        pred_label = tags[pred_tag]

        print(
            f'{word:<15} '
            f'{true_label:<10} '
            f'{pred_label}'
        )
```

## Output

The dataset contains **35,177 unique words** and **17 unique NER tags**.

Training output:

```text
Using device: cuda

Epoch 1: Train Loss=0.1753, Val Loss=0.5360
Epoch 2: Train Loss=0.1349, Val Loss=0.5838
Epoch 3: Train Loss=0.1092, Val Loss=0.6455
```

Classification report:

```text
              precision    recall  f1-score   support

       B-art       0.27      0.18      0.22        96
       B-eve       0.42      0.26      0.32        78
       B-geo       0.86      0.88      0.87      7377
       B-gpe       0.94      0.93      0.94      3176
       B-nat       0.79      0.65      0.71        40
       B-org       0.74      0.77      0.75      3907
       B-per       0.82      0.84      0.83      3362
       B-tim       0.92      0.93      0.93      4104
       I-art       0.19      0.09      0.12        86
       I-eve       0.41      0.33      0.37        70
       I-geo       0.75      0.79      0.77      1448
       I-gpe       0.69      0.55      0.61        33
       I-nat       0.60      0.43      0.50        14
       I-org       0.84      0.79      0.82      3252
       I-per       0.87      0.84      0.86      3423
       I-tim       0.86      0.84      0.85      1341

    accuracy                           0.85     31807
   macro avg       0.69      0.63      0.65     31807
weighted avg       0.85      0.85      0.85     31807
```


<img width="600" height="432" alt="image" src="https://github.com/user-attachments/assets/50c91c29-8a88-4d69-9b8a-520933182979" />

<img width="711" height="377" alt="image" src="https://github.com/user-attachments/assets/abda7088-26be-45a6-869f-992fbf89cf26" />


<img width="672" height="181" alt="image" src="https://github.com/user-attachments/assets/efc418de-1d8d-4099-8755-3fce362f48ec" />



The notebook also produces a **Loss Over Epochs** graph and a word-level comparison of true and predicted NER tags.

## Result

Thus, the Deep Learning model for Named Entity Recognition was developed successfully using a Bidirectional LSTM. The model achieved an overall accuracy of **85%** on the evaluated NER tags.
