import torch
import torchvision
from torch import nn
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader
from skimage import io
import csv
import matplotlib.pyplot as plt


VGG19 = [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256.0,
        "M",
        512,
        512,
        512,
        512.0,
        "M",
        512,
        512,
        512,
        512.0,
        "M",
    ]

class VGG(nn.Module):
    def __init__(
        self,
        architecture,
        in_channels=3, 
        in_height=224, 
        in_width=224, 
        num_hidden=4096,
        num_classes=10
    ):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.in_width = in_width
        self.in_height = in_height
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.convs = self.init_convs(architecture)
        self.fcs = self.init_fcs(architecture)
        
    def forward(self, x):
        x = self.convs(x)
        x = x.reshape(x.size(0), -1)
        x = self.fcs(x)
        return x
    
    def init_fcs(self, architecture):
        pool_count = architecture.count("M")
        factor = (2 ** pool_count)
        if (self.in_height % factor) + (self.in_width % factor) != 0:
            raise ValueError(
                f"`in_height` and `in_width` must be multiples of {factor}"
            )
        out_height = self.in_height // factor
        out_width = self.in_width // factor
        last_out_channels = next(
            x for x in architecture[::-1] if type(x) == int
        )
        return nn.Sequential(
            nn.Linear(
                last_out_channels * out_height * out_width, 
                self.num_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.num_hidden, self.num_classes)
            # nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Linear(self.num_hidden, self.num_classes)
        )
    
    def init_convs(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers.extend(
                    [
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=(3, 3),
                            stride=(1, 1),
                            padding=(1, 1),
                        ),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                    ]
                )
                in_channels = x
            elif type(x) == float:
                out_channels = int(x)
                layers.extend(
                    [
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=(1, 1),
                            stride=(1, 1),
                        ),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                    ]
                )
                in_channels = int(x)
            else:
                layers.append(
                    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
                )

        return nn.Sequential(*layers)

class SportLoader(Dataset):
    def __init__(self, mode, img_list, label_list=None, transform=torchvision.transforms.ToTensor()):
        self.mode = mode
        self.img_list = img_list  # imd_name list from csv file
        self.label_list = label_list  # label list from csv file
        self.transform = transform
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = "dataset/" + self.mode + "/" + self.img_list[index]
        img = io.imread(img_path)
        img = self.transform(img)
        label = torch.tensor(int(self.label_list[index]))
        return img, label 

def get_data(mode):
    img_list = []
    label_list = []

    with open("dataset/"+ mode +".csv", 'r') as file:
        csvreader = csv.reader(file)
        next(csvreader)
        for row in csvreader:
            img_list.append(row[0])
            label_list.append(row[1])
    return img_list, label_list

def plot(num_epoch, train, val, title):
    plt.plot(range(1, num_epoch+1), train, '-b', label='train')
    plt.plot(range(1, num_epoch+1), val, '-r', label='val')

    plt.xlabel("n epoch")
    plt.legend(loc='upper left')
    plt.title(title)

    # save image
    plt.savefig(title+".png")  # should before show method

    # show
    plt.show()
    plt.clf()


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = VGG19 = VGG(in_channels=3, 
    in_height=224, 
    in_width=224, 
    architecture=VGG19).to(device)


train_img, train_label = get_data('train')
val_img, val_label = get_data('val')

train_dataset = SportLoader("train", train_img, train_label)
val_dataset = SportLoader("val", val_img, val_label)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

num_epochs = 50
learning_rate = 0.005

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  


# Train the model
total_step = len(train_loader)

train_loss_list = []
val_loss_list = []
train_acc = []
val_acc = []
for epoch in range(num_epochs):
    train_correct = 0
    train_total = 0
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        train_loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # statistc
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()
        train_total += labels.size(0)

    print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {train_loss.item():.4f}')

    # Validation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            val_loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
    
        print(f'Accuracy of the network on the {total} validation images: {100 * correct / total} %')

    train_loss_list.append(train_loss.item())
    val_loss_list.append(val_loss.item())
    train_acc.append(100 * train_correct / train_total)
    val_acc.append(100 * correct / total)


plot(num_epochs, train_loss_list, val_loss_list, "Loss Curve")
plot(num_epochs, train_acc, val_acc, "Accuracy Curve")

# get number of parameters
num_of_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"number of parameters: {num_of_parameters}")

torch.save(model, "HW1_311554021.pt")
# 122840906 VGG19 - 1FC
# 118122314 VGG19 - 1FC with kernel size == 1 before last3 max pooling