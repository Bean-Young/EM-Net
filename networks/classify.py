import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        return out + x 

class AttentionModel(nn.Module):
    def __init__(self, in_dim1, in_dim2, n_doms):
        super(AttentionModel, self).__init__()
        self.conv1_x = nn.Conv2d(in_channels=in_dim1[1], out_channels=64, kernel_size=3, padding='same')
        self.bn1_x = nn.BatchNorm2d(64)
        self.relu1_x = nn.ReLU()
        self.pool1_x = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attention1_x = SelfAttention(in_dim=64)

        self.conv2_x = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.bn2_x = nn.BatchNorm2d(128)
        self.relu2_x = nn.ReLU()
        self.pool2_x = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attention2_x = SelfAttention(in_dim=128)

        self.conv3_x = nn.Conv2d(128, 256, kernel_size=3, padding='same')
        self.bn3_x = nn.BatchNorm2d(256)
        self.relu3_x = nn.ReLU()
        self.pool3_x = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attention3_x = SelfAttention(in_dim=256)

        self.conv4_x = nn.Conv2d(256, 512, kernel_size=3, padding='same')
        self.bn4_x = nn.BatchNorm2d(512)
        self.relu4_x = nn.ReLU()
        self.pool4_x = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attention4_x = SelfAttention(in_dim=512)

        self.conv1_y = nn.Conv2d(in_channels=in_dim2[1], out_channels=256, kernel_size=3, padding='same')
        self.bn1_y = nn.BatchNorm2d(256)
        self.relu1_y = nn.ReLU()
        self.attention1_y = SelfAttention(in_dim=256)

        self.conv_cat = nn.Conv2d(512 + 256, 512, kernel_size=3, padding='same')
        self.bn_cat = nn.BatchNorm2d(512)
        self.relu_cat = nn.ReLU()

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(512 * 14 * 14, 1024)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc_final = nn.Linear(128, n_doms)

    def forward(self, inputx, inputy):
        x = self.pool1_x(self.relu1_x(self.bn1_x(self.conv1_x(inputx))))

        x = self.pool2_x(self.relu2_x(self.bn2_x(self.conv2_x(x))))
        x = self.pool3_x(self.relu3_x(self.bn3_x(self.conv3_x(x))))
        x = self.pool4_x(self.relu4_x(self.bn4_x(self.conv4_x(x))))
        x = self.attention4_x(x)
        y = self.relu1_y(self.bn1_y(self.conv1_y(inputy)))
        y = self.attention1_y(y)

        combined = torch.cat((x, y), dim=1)
        combined = self.relu_cat(self.bn_cat(self.conv_cat(combined)))

        combined = self.flatten(combined)
        combined = F.relu(self.fc1(combined))
        combined = self.dropout1(combined)
        combined = F.relu(self.fc2(combined))
        combined = self.dropout2(combined)
        output = self.fc_final(combined)

        return output
