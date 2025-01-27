from utils import *

class TransformerBlock(nn.Module):
    def __init__(self, in_channels, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(in_channels).cuda()
        self.attention = nn.MultiheadAttention(in_channels, num_heads).cuda()
        self.layer_norm2 = nn.LayerNorm(in_channels).cuda()
        self.feed_forward = nn.Sequential(
            nn.Linear(in_channels, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, in_channels),
        ).cuda()
        self.initialize_identity(in_channels)

    def forward(self, x):
        x = self.layer_norm1(x)
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output

        x = self.layer_norm2(x)
        x = x + self.feed_forward(x)

        return x

    def initialize_identity(self, in_channels):
        # Initialize attention weights to identity mapping
        for name, param in self.attention.named_parameters():
            if "weight" in name:
                nn.init.eye_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

        nn.init.eye_(self.feed_forward[0].weight)  # First Linear layer
        nn.init.constant_(self.feed_forward[0].bias, 0)
        nn.init.eye_(self.feed_forward[2].weight)  # Second Linear layer
        nn.init.constant_(self.feed_forward[2].bias, 0)

        # Initialize layer normalization to be a no-op
        nn.init.constant_(self.layer_norm1.weight, 1)
        nn.init.constant_(self.layer_norm1.bias, 0)
        nn.init.constant_(self.layer_norm2.weight, 1)
        nn.init.constant_(self.layer_norm2.bias, 0)


class TransformerModel(nn.Module):
    def __init__(self, num_blocks, num_past_images, num_heads, num_fc_layers, phase_correlation=True, past_images=True):
        super(TransformerModel, self).__init__()

        self.num_blocks = num_blocks
        self.num_past_images = num_past_images
        self.num_heads = num_heads
        self.num_fc_layers = num_fc_layers

        if phase_correlation and num_past_images > 0:
            self.in_channels_transformer = (384 + 2) * self.num_past_images + 384
            self.in_channels_fc = (384 + 2) * self.num_past_images
        elif num_past_images > 0:
            self.in_channels_transformer = 384 * self.num_past_images + 384
            self.in_channels_fc = 384 * self.num_past_images
        else:
            self.in_channels_transformer = 384 * self.num_past_images + 384
            self.in_channels_fc = 384
        self.out_channels = 384

        self.block_list = []

        for i in range(4):
            self.block_list.append(nn.ModuleList())
            for _ in range(num_blocks):
                self.block_list[i].append(TransformerBlock(self.in_channels_transformer, self.num_heads, self.in_channels_transformer))
            self.block_list[i].append(nn.Linear(self.in_channels_transformer, self.out_channels).cuda())

        self.fclist = []
        for i in range(self.num_fc_layers):
            self.fclist.append(nn.ModuleList([
                nn.Linear(self.in_channels_fc, self.in_channels_fc)
                for _ in range(4)
            ]).cuda())
        self.fc_out = nn.ModuleList([
            nn.Linear(self.in_channels_fc, self.out_channels)
            for _ in range(4)
        ]).cuda()
        self._initialize_weights()

    def _initialize_weights(self):
        for layer_idx in range(len(self.fclist)):
            for i, fc in enumerate(self.fclist[layer_idx]):
                with torch.no_grad():
                    init.constant_(fc.bias, 0)
                    fc.weight.zero_()

        for i, fc in enumerate(self.fc_out):
            with torch.no_grad():
                init.constant_(fc.bias, 0)
                fc.weight.zero_()

        for i in range(4):
            with torch.no_grad():
                init.constant_(self.block_list[i][-1].bias, 0)
                self.block_list[i][-1].weight.zero_()

    def forward_transformer(self, x, idx):
        x_flat = x.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])
        x_flat = x_flat.permute(2, 0, 1)

        for i in range(len(self.block_list[idx]) - 1):
            x_flat = self.block_list[idx][i](x_flat)
        x_flat = self.block_list[idx][-1](x_flat)

        x_flat = x_flat.permute(1, 2, 0)
        x = x[:, :384] + x_flat.view(x.shape[0], self.out_channels, x.shape[2], x.shape[3])

        return x

    def forward_fc(self, x, idx):
        for layer_idx in range(len(self.fclist)):
            if self.num_past_images > 0:
               x_out = self.fclist[layer_idx][idx](x[:, 384:])
            else:
               x_out = self.fclist[layer_idx][idx](x)
            x_out = torch.relu(x_out)
        x_out = self.fc_out[idx](x_out) + x[:, :384]
        return x_out
