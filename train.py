import torch
from tqdm import tqdm  # Import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from get_loader import get_loader
from model import CNNtoRNN
from utils import save_checkpoint, load_checkpoint, print_examples


def train():
    transform = transforms.Compose([
        transforms.Resize((356, 356)),
        transforms.RandomCrop((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_loader, dataset = get_loader(
        root_folder="/home/eba/Dataset/exp/test_img",
        annotation_file="/home/eba/Dataset/exp/annotations.csv",
        transform=transform,
        num_workers=2,
    )

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = False
    save_model = True
    train_CNN = False

    # Hyperparameters
    embed_size = 768
    hidden_size = 512
    vocab_size = 32000
    num_layers = 2
    learning_rate = 3e-4
    num_epochs = 100

    # for tensorboard
    writer = SummaryWriter("runs/flickr")
    step = 0

    # initialize model, loss etc
    model = CNNtoRNN(hidden_size, num_layers, embed_size, vocab_size, device).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming <PAD> token index is 0
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Only finetune the CNN
    for name, param in model.encoderCNN.inception.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN

    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    model.train()

    for epoch in range(num_epochs):
        # Save the model at the start of the epoch

        print_examples(model, device, dataset)

        if save_model:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            save_checkpoint(checkpoint)

        # Training loop
        for idx, (imgs, captions) in tqdm(
            enumerate(train_loader), total=len(train_loader), leave=False
        ):
            imgs = imgs.to(device)
            captions = captions.to(device)

            # Adjust for the last batch which might be smaller
            current_batch_size = imgs.size(0)
            captions = captions[:current_batch_size, :]

            # Zero the gradients before running the forward pass.
            optimizer.zero_grad()

            # Forward pass
            outputs = model(imgs, captions)

            # Calculate the loss
            # The output of the model is expected to be [current_batch_size, sequence_length, vocab_size]
            # and the captions are [current_batch_size, sequence_length]
            # We reshape them to be [current_batch_size * sequence_length, vocab_size] and [current_batch_size * sequence_length] respectively
            # We also need to exclude the first column of the outputs, which corresponds to the <SOS> token
            outputs = outputs[:, 1:, :].reshape(-1, outputs.shape[2])
            targets = captions[:, 1:].reshape(-1)

            # Compute the loss
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            
            optimizer.step()
            #optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
            #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)


            # Update the step and log the loss
            step += 1
            writer.add_scalar("Training loss", loss.item(), global_step=step)

            # Print the loss for the current batch
            print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")

if __name__ == "__main__":
    train()
