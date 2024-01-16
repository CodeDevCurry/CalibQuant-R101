import os
import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_imagenet_test_data(path: str):
    """
    Load the ImageNet test dataset.

    Parameters:
    path (str): Path to the ImageNet dataset folder.

    Returns:
    DataLoader: DataLoader for the ImageNet test dataset.
    """
    # Fixed parameters
    input_size = 224
    batch_size = 64
    num_workers = 4
    pin_memory = True
    test_resize = 256

    # Normalization transform
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Test dataset directory
    # Assuming 'val' folder contains test data
    testdir = os.path.join(path, 'val')

    # Test dataset with transforms
    test_dataset = datasets.ImageFolder(testdir, transforms.Compose([
        transforms.Resize(test_resize),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize,
    ]))

    # DataLoader for the test dataset
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return test_loader


@torch.no_grad()
def validate_model(val_loader, model, device=None, print_freq=100):
    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)

    test_loss = 0
    total = 0
    correct = 0
    correct_5 = 0

    model.eval()
    for i, (images, target) in enumerate(val_loader):
        images = images.to(device)
        target = target.to(device)
        # compute output
        output = model(images)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        test_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        _, predicted_5 = output.topk(5, 1, True, True)
        predicted_5 = predicted_5.t()
        correct_ = predicted_5.eq(target.view(1, -1).expand_as(predicted_5))
        correct_5 += correct_[:5].reshape(-1).float().sum(0,
                                                          keepdim=True).item()
        if i % 10 == 0 or i == len(val_loader) - 1:
            print('test: [batch: %d/%d ] | Loss: %.3f | Acc: %.3f%% (%d/%d)/ %.3f%% (%d/%d)'
                  % (i, len(val_loader), test_loss/(i+1), 100.*correct/total, correct, total, 100.*correct_5/total, correct_5, total))
    acc = 100.*correct/total
    print("Final accuracy: %.3f" % acc)
    return acc


# Model
print('==> Building quantized model..')

# Load validation data
print('==> Preparing data..')
data_path = "/home/u2208283129/CalibQuant-R101/imagenet"

val_loader = load_imagenet_test_data(data_path)

model_path = "/home/u2208283129/CalibQuant-R101/model_zoo/resnet101_imagenet.pth"
model = models.resnet101(pretrained=False)

# Load model weights
model.load_state_dict(torch.load(model_path))

print('==> Validate Accuracy..')
validate_model(val_loader, model, device)
