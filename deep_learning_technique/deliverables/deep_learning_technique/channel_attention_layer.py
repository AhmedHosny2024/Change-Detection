import torch 
import torch.nn as nn
class ChannelAttention(nn.Module):
    def __init__(self, channel):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.query_conv = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=1),
            )
        self.key_conv = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=1),
            )
        self.value_conv = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=channel, kernel_size=1),
            )
        self.sigmoid = nn.Sigmoid()
        self.sm = nn.Softmax(-1)

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        query = self.query_conv(y).transpose(-1, -2)
        key = self.key_conv(y)
        attention = torch.bmm(query, key)
        attention = self.sm(attention)
        value = self.value_conv(y)
        out = torch.bmm(value, attention).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        x = x * out.expand_as(x)
        return x
# test the channel attention layer
if __name__ == "__main__":
    channel_attention = ChannelAttention(1)
    x = torch.randn(1, 3, 256, 256)
    y = channel_attention(x)
    print(y.shape)