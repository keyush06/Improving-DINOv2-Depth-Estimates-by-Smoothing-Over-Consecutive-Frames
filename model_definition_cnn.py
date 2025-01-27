from utils import *

class CNNModel(nn.Module):
    def __init__(self, num_layers, num_past_images, phase_correlation=True, past_images=True):
        super(CNNModel, self).__init__()
        self.num_layers = num_layers
        self.num_past_images = num_past_images
        if phase_correlation and num_past_images > 0:
            in_channels = (384 + 2) * self.num_past_images
        elif num_past_images > 0:
            in_channels = 384 * self.num_past_images
        else:
            in_channels = 384
        out_channels = 384
        self.convlist = []
        for i in range(self.num_layers):
            self.convlist.append(nn.ModuleList([
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
                for _ in range(4)
            ]).cuda())
        self.conv_out = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
            for _ in range(4)
        ]).cuda()
        self.fclist = []
        for i in range(self.num_layers):
            self.fclist.append(nn.ModuleList([
                nn.Linear(in_channels, in_channels)
                for _ in range(4)
            ]).cuda())
        self.fc_out = nn.ModuleList([
            nn.Linear(in_channels, out_channels)
            for _ in range(4)
        ]).cuda()
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize convolutional layers
        for layer_idx in range(len(self.convlist)):
            for i, conv in enumerate(self.convlist[layer_idx]):
                with torch.no_grad():
                    init.constant_(conv.bias, 0)
                    conv.weight.zero_()

        for i, conv in enumerate(self.conv_out):
            with torch.no_grad():
                init.constant_(conv.bias, 0)
                conv.weight.zero_()

        # Initialize fully connected layers
        for layer_idx in range(len(self.fclist)):
            for i, fc in enumerate(self.fclist[layer_idx]):
                with torch.no_grad():
                    init.constant_(fc.bias, 0)
                    fc.weight.zero_()

        for i, fc in enumerate(self.fc_out):
            with torch.no_grad():
                init.constant_(fc.bias, 0)
                fc.weight.zero_()

    def forward_conv(self, x, idx):
        for layer_idx in range(len(self.convlist)):
            if self.num_past_images > 0:
                x_out = self.convlist[layer_idx][idx](x[:, 384:])
            else:
                x_out = self.convlist[layer_idx][idx](x)
            x_out = torch.relu(x_out)
        x_out = self.conv_out[idx](x_out) + x[:, :384]
        return x_out

    def forward_fc(self, x, idx):
        for layer_idx in range(len(self.fclist)):
            if self.num_past_images > 0:
               x_out = self.fclist[layer_idx][idx](x[:, 384:])
            else:
               x_out = self.fclist[layer_idx][idx](x)
            x_out = torch.relu(x_out)
        x_out = self.fc_out[idx](x_out) + x[:, :384]
        return x_out
