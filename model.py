import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel
from torchvision.models import inception_v3, Inception_V3_Weights

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        # Get the main output from the Inception model
        features = self.inception(images)
        return self.dropout(self.relu(features))


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 32000)  # Make sure this matches the vocab size
        self.dropout = nn.Dropout(0.5)
        self.features_to_hidden = nn.Linear(embed_size, hidden_size)  # Linear layer to adapt feature size

    def forward(self, features, captions):
        self.lstm.flatten_parameters()
        # Transform features to the correct hidden state size
        features = self.features_to_hidden(features)

        # Now features should have the same size as the hidden size expected by the LSTM
        hidden_state = features.unsqueeze(0).expand(self.lstm.num_layers, *features.size())
        cell_state = torch.zeros_like(hidden_state)

        # Forward pass through LSTM
        hiddens, _ = self.lstm(captions, (hidden_state, cell_state))

        # Pass the output of the LSTM to the linear layer
        outputs = self.dropout(self.linear(hiddens))
        return outputs


class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, num_layers)
        self.bert = BertModel.from_pretrained('tugstugi/bert-base-mongolian-cased')

    def forward(self, images, captions, attention_mask=None):
        features = self.encoderCNN(images)

        # Get the BERT embeddings for the captions
        bert_output = self.bert(captions, attention_mask=attention_mask).last_hidden_state

        # Pass both the features and the BERT embeddings to the DecoderRNN
        outputs = self.decoderRNN(features, bert_output)
        return outputs

    def caption_image(self, image, tokenizer, max_length=50):
        result_caption = []

        with torch.no_grad():
            features = self.encoderCNN(image).unsqueeze(0)
            states = None  # LSTM states are initially None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(features, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                predicted_word_idx = predicted.item()
                result_caption.append(predicted_word_idx)

                # Prepare the next input token for the LSTM
                next_token = torch.tensor([[predicted_word_idx]], device=image.device)
                bert_output = self.bert(next_token).last_hidden_state
                features = bert_output

                if predicted_word_idx == tokenizer.eos_token_id:
                    break

        return [tokenizer.decode([idx], skip_special_tokens=True) for idx in result_caption]

