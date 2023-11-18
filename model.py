import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import Inception_V3_Weights

# Assuming BERT's embedding size is 768. Verify and adjust if needed.
BERT_EMBEDDING_SIZE = 768
BERT_VOCAB_SIZE = 32000  # Mongolian BERT's vocabulary size


class EncoderCNN(nn.Module):
    def __init__(self, train_CNN=False, embed_size=BERT_EMBEDDING_SIZE):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,
                                             transform_input=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features,
                                      embed_size)
        self.inception.AuxLogits = None
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images)
        if isinstance(features, tuple):
            features = features[0] # Extract the main output
        return self.dropout(self.relu(features))


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, num_layers, embed_size=BERT_EMBEDDING_SIZE, vocab_size=BERT_VOCAB_SIZE):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)  # First vocab_size, then embed_size
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        # Check captions input just before embedding
        #print("Captions input to embed:", captions)  # Add this line

        # Embed the captions
        embeddings = self.embed(captions)  # [batch size, caption length, embed size]
        #print("Embeddings shape:", embeddings.shape)

        # Check the shape of the features tensor
        #print("Features shape before unsqueeze:", features.shape)
        features = features.unsqueeze(1)#.repeat(1, captions.size(1), 1)  # [batch size, 1, embed size]
        #print("Features shape after unsqueeze:", features.shape)

        # Ensure that features and captions have the same batch size
        batch_size = features.size(0)
        captions = captions[:batch_size, :]

        # Concatenate the image features and embeddings
        # Now combined should be [batch size, 1 + caption length, embed size]
        combined = torch.cat((features, embeddings), dim=1)
        print("Combined shape:", combined.shape)

        # Pass the combined tensor to the LSTM
        lstm_out, _ = self.lstm(combined)
        print("LSTM output shape:", lstm_out.shape)

        # Take all LSTM outputs except the first one which corresponds to the image feature
        lstm_out = lstm_out[:, 1:, :]  # [batch size, caption length, hidden size]

        # Dropout and linear layer
        outputs = self.dropout(lstm_out)
        outputs = self.linear(outputs)
        print("Outputs shape:", outputs.shape)

        return outputs



class CNNtoRNN(nn.Module):
    def __init__(self, hidden_size, num_layers, embed_size=768, vocab_size=32000, device="cpu"):
        super(CNNtoRNN, self).__init__()
        self.device = device
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(hidden_size, num_layers, embed_size, vocab_size)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        #print("Image features shape:", features.shape)
        #print("Captions shape:", captions.shape)

        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                predicted_item = predicted.item()

                # Check if the predicted token ID is in the vocabulary
                if predicted_item in vocabulary.itos:
                    token = vocabulary.itos[predicted_item]
                else:
                    token = "<UNK>"  # Use a default unknown token

                result_caption.append(token)

                # Convert the token back to an index for the next prediction step
                # Handle the case where the token might be unknown
                if token in vocabulary.stoi:
                    next_input = vocabulary.stoi[token]
                else:
                    next_input = vocabulary.stoi["<UNK>"]

                x = self.decoderRNN.embed(torch.tensor([next_input], device=self.device)).unsqueeze(0)

                if token == "<EOS>":
                    break

        return result_caption
