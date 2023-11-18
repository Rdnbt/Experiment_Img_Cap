import torch
import torchvision.transforms as transforms
from PIL import Image


def print_examples(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    model.eval()
    test_img1 = transform(Image.open("/home/eba/Research/ML_experiment/test/dogandguy.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 1 CORRECT: Саарал цамцтай залуу болон нохой орон дээр унтаж байна")
    print(
        "Example 1 OUTPUT: "
        + " ".join(model.caption_image(test_img1.to(device), dataset.vocab))
    )
    test_img2 = transform(
        Image.open("/home/eba/Research/ML_experiment/test/children.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 2 CORRECT: Хоёр хүүхэд газарт сууж тоглож байна")
    print(
        "Example 2 OUTPUT: "
        + " ".join(model.caption_image(test_img2.to(device), dataset.vocab))
    )
    test_img3 = transform(Image.open("/home/eba/Research/ML_experiment/test/party.jpg").convert("RGB")).unsqueeze(
        0
    )
    print("Example 3 CORRECT: Эрэгтэй эмэгтэй хоёр компьютер харан инээлдэж байна")
    print(
        "Example 3 OUTPUT: "
        + " ".join(model.caption_image(test_img3.to(device), dataset.vocab))
    )
    test_img4 = transform(
        Image.open("/home/eba/Research/ML_experiment/test/redguy.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 4 CORRECT: Улаан саарал цамцтай залуу гартаа юм барин алхаж байна")
    print(
        "Example 4 OUTPUT: "
        + " ".join(model.caption_image(test_img4.to(device), dataset.vocab))
    )
    test_img5 = transform(
        Image.open("/home/eba/Research/ML_experiment/test/smile.jpg").convert("RGB")
    ).unsqueeze(0)
    print("Example 5 CORRECT: Хоол хийж буй эрэгтэй хүүхэд инээмсэглэж байна")
    print(
        "Example 5 OUTPUT: "
        + " ".join(model.caption_image(test_img5.to(device), dataset.vocab))
    )
    model.train()


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step
