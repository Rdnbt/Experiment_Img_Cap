import torch
from torchvision.transforms import transforms
from model import CNNtoRNN
from get_loader import FlickrDataset, MyCollate, tokenizer

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the transform
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        # Make sure to include the normalization used during training
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create the dataset
    dataset = FlickrDataset(
        root_dir='/home/eba/Dataset/exp/test_50',  # replace with your image directory
        captions_file='/home/eba/Dataset/exp/50_annotations.csv',  # replace with your captions file
        transform=transform
    )

    # Select a sample for testing
    img, caption = dataset[0]  # for example, testing the first sample

    # Initialize model
    embed_size = 256
    hidden_size = 512
    num_layers = 2
    model = CNNtoRNN(embed_size, hidden_size, num_layers).to(device)
    model.eval()  # set the model to evaluation mode

    # Process the image and caption to be suitable for the model input
    img = img.to(device).unsqueeze(0)  # Add batch dimension
    caption = caption.to(device).unsqueeze(1)  # Add batch dimension

    # Forward pass through the model
    with torch.no_grad():
        output = model(img, caption[:-1])  # exclude the <EOS> token for the input

    print("Model output shape:", output.shape)

    # Optionally, generate a caption for the image
    generated_caption = model.caption_image(img, tokenizer, max_length=50)
    print("Generated caption:", ' '.join(generated_caption))

if __name__ == "__main__":
    test_model()

