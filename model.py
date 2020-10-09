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

    @staticmethod
    def preprocess(image):
        image = image[70:135, :]
        # cv2.imshow("cropped", image)
        # cv2.waitKey(0)
        # image = cv2.resize(image, (200, 66))
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
        image = SdcSimDataset.preprocess(image)
        images.append(image.transpose(2,0,1))
        angles.append(angle)

        # Augmenting images by flipping across y-axis
        images.append(cv2.flip(image, 1).transpose(2,0,1))
        angles.append(-angle)

        # LEFT
        image = self.get_image(line[1], self.root_path)
        image = SdcSimDataset.preprocess(image)
        images.append(image.transpose(2,0,1))
        angles.append(angle + correction)

        # RIGHT
        image = self.get_image(line[2], self.root_path)
        image = SdcSimDataset.preprocess(image)
        images.append(image.transpose(2,0,1))
        angles.append(angle - correction)

        return images, angles

# TODO
# create the network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

    def forward(self, x):
        pass

def collate_fn(batch):
    images = []
    angles = []
    for each in batch:
        images.extend(each[0])
        angles.extend(each[1])
    return torch.from_numpy(np.stack(images)).float(), torch.from_numpy(np.stack(angles)).float().unsqueeze(-1)

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    # TODO
    # change path according to your system
    dataset = SdcSimDataset('/home/ridhwan/Downloads/beta_simulator_linux/recording_data/')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn)
    net = ConvNet()
    if torch.cuda.is_available():
        net = net.cuda()
    # TODO
    # Create loss and optimizer
    criterion =
    optimizer =
    for epoch in range(5):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (images, angles) in enumerate(dataloader):
            if torch.cuda.is_available():
                images = images.cuda()
                angles = angles.cuda()

            # TODO
            # add code for forward pass, loss calc, back pass and optimizer step

            # print statistics
            running_loss += loss.item()
            print_every = 10
            if i % print_every == print_every - 1:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / print_every))
                running_loss = 0.0

    torch.save(net, 'model.pkl')