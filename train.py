import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model.model import KernelTransformer, MaskedKernelTransformer
from csv_logger import log_csv
from tqdm import tqdm


save_model = True
continue_training = False


def save_checkpoint(state, filename='checkpoint/checkpoint.pth.tar'):
    print('saving model checkpoint ...')
    torch.save(state, filename)


# Data augmentation and normalization
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=512,
                                          shuffle=True, num_workers=2)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=512,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
           'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(' - Training device currently set to:', device)

model = MaskedKernelTransformer(in_channels=3, emb_size=96, patch_size=2, 
                                heads=8, num_classes=10, struct=(2, 2, 6, 2), mask_ratio=0.1).to(device)
# model = KernelTransformer(in_channels=3, emb_size=96, patch_size=2, 
#                           heads=8, num_classes=10, struct=(2, 2, 6, 2)).to(device)
model = nn.DataParallel(model)
criterion = torch.nn.CrossEntropyLoss()
num_epochs = 300
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4,
#                                                 steps_per_epoch=len(train_loader),
#                                                 epochs=num_epochs,
#                                                 pct_start=0.05)

start_epoch = 0
if continue_training:
    checkpoint = torch.load('checkpoint/checkpoint.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['epoch']
    print('checkpoint loaded, resuming on epoch: ', start_epoch)

for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    bar = tqdm(total=len(train_loader))
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images, masked=True)
        # calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        bar.update(1)

    bar.close()

    train_accuracy = round((correct / total) * 100, 2)
    print(f'Training Accuracy: {train_accuracy}')

    epoch_loss = running_loss / len(train_set)
    print(f'Loss: {epoch_loss}')

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        bar = tqdm(total=len(test_loader))
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images, masked=False)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            bar.update(1)

        bar.close()

    accuracy = round((correct / total) * 100, 2)
    print(f'Val. Accuracy: {accuracy}')

    if save_model:
        save_checkpoint({
            'epoch': epoch + 1,
            'accuracy': accuracy,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        })
    log_csv(epoch, train_accuracy, accuracy, epoch_loss)

    scheduler.step()

print('Finished Training')
