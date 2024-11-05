import torch
from torch.nn import functional as F
from torch import nn

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(self, transformer_dim: int, transformer: nn.Module, numClasses) -> None:
        super().__init__()
        self.numClasses = numClasses
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.up1 = Up(in_channel=transformer_dim, out_channel=transformer_dim//2)
        self.up2 = Up(in_channel=transformer_dim//2, out_channel=transformer_dim//4)
        self.up3 = Up(in_channel=transformer_dim // 4, out_channel=transformer_dim // 8)
        self.up4 = Up(in_channel=transformer_dim // 8, out_channel=transformer_dim // 16)

        self.mask_tokens = nn.Embedding(numClasses, transformer_dim)
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 16, 3)
                for i in range(numClasses)
            ]
        )

    def forward(
            self,
            image_embeddings,
            pe: torch.Tensor,
            prompt_embeddings,
            prompt_skip,
    ):

        b, c, h, w = image_embeddings.shape
        pe = torch.repeat_interleave(pe, b, dim=0)

        mask_tokens = self.mask_tokens.weight
        mask_tokens = mask_tokens.unsqueeze(0).expand(prompt_embeddings.shape[0], -1, -1)
        prompt_tokens = prompt_embeddings.flatten(2).permute(0, 2, 1)
        tokens = torch.cat([mask_tokens, prompt_tokens], dim=1)

        # Run the transformer
        hs, src = self.transformer(image_embedding=image_embeddings, prompt_embedding=tokens, image_pe=pe)

        src = src.transpose(1, 2).view(b, c, h, w)
        mask_tokens_out = hs[:, :self.numClasses, :]

        upscaled_embedding = self.up1(src, prompt_skip[0])
        upscaled_embedding = self.up2(upscaled_embedding, prompt_skip[1])
        upscaled_embedding = self.up3(upscaled_embedding, prompt_skip[2])
        upscaled_embedding = self.up4(upscaled_embedding, prompt_skip[3])

        hyper_in_list = []
        for i in range(self.numClasses):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        mask = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        return mask



class Double_Conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Double_Conv, self).__init__()

        mid_channel = in_channel//2 if in_channel > out_channel else out_channel//2

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=mid_channel, kernel_size=3, stride=1, padding=1),
            LayerNorm2d(mid_channel),
            nn.GELU(),

            nn.Conv2d(in_channels=mid_channel, out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            LayerNorm2d(out_channel),
            nn.GELU(),
        )

    def forward(self,x):
        return self.double_conv(x)


class Up(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channel, in_channel, kernel_size=2, stride=2),
            LayerNorm2d(in_channel),
            nn.GELU(),
        )
        self.double_conv = Double_Conv(in_channel=in_channel+in_channel//2, out_channel=out_channel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        mid_x = torch.cat([x1, x2], dim=1)
        return self.double_conv(mid_x)



class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
