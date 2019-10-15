import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import csv
import cv2
import numpy as np

class SdcSimDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = root_path
        self.lines = []
        with open(root_path + 'driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                self.lines.append(line)

    def __len__(self):
        return len(self.lines)

    def get_image(self, path, base_path='../data/'):
        # load image and conver to RGB
        filename = path.split('/')[-1]
        path = base_path + 'IMG/' + filename
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def preprocess(self, image):
        image = image[70:135, :]
        # cv2.imshow("cropped", image)
        # cv2.waitKey(0)
        # image = cv2.resize(image, (32, 32))
        image = image / 255.0 - 0.5
        return image

    def __getitem__(self, idx):
        images = []
        angles = []
        line = self.lines[idx]
        angle = float(line[3])
        correction = 0.2

        # CENTER
        image = self.get_image(line[0], self.root_path)
        image = self.preprocess(image)
        images.append(image.transpose(2,0,1))
        angles.append(angle)

        # Augmenting images by flipping across y-axis
        images.append(cv2.flip(image, 1).transpose(2,0,1))
        angles.append(-angle)

        # LEFT
        image = self.get_image(line[1], self.root_path)
        image = self.preprocess(image)
        images.append(image.transpose(2,0,1))
        angles.append(angle + correction)

        # RIGHT
        image = self.get_image(line[2], self.root_path)
        image = self.preprocess(image)
        images.append(image.transpose(2,0,1))
        angles.append(angle - correction)

        # X_train = torch.Tensor(np.stack(images))
        # y_train = torch.Tensor(angles)
        # breakpoint()
        # sample = {'image': torch.from_numpy(np.stack(images)),
        #         'angles': torch.from_numpy(np.stack(angles))}
        sample = (images, angles)

        return sample

class NvidiaNet(nn.Module):
    def __init__(self):
        super(NvidiaNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 24, 5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, 5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, 5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, 3)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.fc1   = nn.Linear(2112, 1164)
        self.fc2   = nn.Linear(1164, 100)
        self.fc3   = nn.Linear(100, 50)
        self.fc4   = nn.Linear(50, 10)
        self.fc5   = nn.Linear(10, 1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.dropout(out, p=0.3)
        out = F.relu(self.fc2(out))
        out = F.dropout(out, p=0.3)
        out = F.relu(self.fc3(out))
        out = F.relu(self.fc4(out))
        out = self.fc5(out)
        return out

def collate_fn(batch):
    images = []
    angles = []
    for each in batch:
        images.extend(each[0])
        angles.extend(each[1])
    return torch.from_numpy(np.stack(images)).float(), torch.from_numpy(np.stack(angles)).float().unsqueeze(-1)

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    dataset = SdcSimDataset('/home/ridhwan/Downloads/beta_simulator_linux/recording_data/')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn)
    net = NvidiaNet()
    if torch.cuda.is_available():
        net = net.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(15):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (images, angles) in enumerate(dataloader):
            if torch.cuda.is_available():
                images = images.cuda()
                angles = angles.cuda()

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, angles)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print_every = 10
            if i % print_every == print_every - 1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / print_every))
                running_loss = 0.0

    torch.save(net, 'model.pkl')