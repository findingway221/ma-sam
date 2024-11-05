from functools import partial

from torch import nn
from torch.nn import functional as F
import torch

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .positional_embedding import PositionalEmbedding
from .prompt_encoder import PromptEncoder
from .transformer import TwoWayTransformer


class MASAM(nn.Module):
    def __init__(self, num_classes, num_atlas, encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
                 encoder_global_attn_indexes=(2, 5, 8, 11), image_size=512, prompt_embed_dim=256, vit_patch_size=16,
                 adapterTrain=True):

        super().__init__()
        image_embedding_size = image_size // vit_patch_size

        self.image_encoder = ImageEncoderViT(
                depth=encoder_depth,
                embed_dim=encoder_embed_dim,
                img_size=image_size,
                in_chans=3,
                mlp_ratio=4,
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
                num_heads=encoder_num_heads,
                patch_size=vit_patch_size,
                qkv_bias=True,
                use_rel_pos=True,
                global_attn_indexes=encoder_global_attn_indexes,
                window_size=14,
                out_chans=prompt_embed_dim,
                adapter_train=adapterTrain
            )

        self.prompt_encoder = PromptEncoder(
                in_channel=num_atlas,
                out_channel=prompt_embed_dim,
            )

        self.mask_decoder = MaskDecoder(
                transformer_dim=prompt_embed_dim,
                numClasses=num_classes,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
            )

        self.pe = PositionalEmbedding(
                embed_dim=prompt_embed_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size)
            )
    def forward(self, img, prompt):

        imgFeatures = self.image_encoder(img)
        promptFeatures, skip = self.prompt_encoder(prompt)
        mask = self.mask_decoder(image_embeddings=imgFeatures,
                                pe=self.pe(),
                                prompt_embeddings=promptFeatures,
                                prompt_skip=skip)

        return mask


def froze(net):
    for n, value in net.module.image_encoder.named_parameters():
        if "Adapter" in n:
            value.requires_grad = True
        else:
            value.requires_grad = False



def load_from(net_dict, state_dicts, image_size, vit_patch_size):
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    new_state_dict = {k: v for k, v in state_dicts.items() if
                      k in net_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[2] not in k}
    pos_embed = new_state_dict['image_encoder.pos_embed']
    token_size = int(image_size // vit_patch_size)
    if pos_embed.shape[1] != token_size:

        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        new_state_dict['image_encoder.pos_embed'] = pos_embed
        rel_pos_keys = [k for k in net_dict.keys() if 'rel_pos' in k]

        global_rel_pos_keys = [k for k in rel_pos_keys if
                                                        '2' in k or
                                                        '5' in k or
                                                        '7' in k or
                                                        '8' in k or
                                                        '11' in k or
                                                        '13' in k or
                                                        '15' in k or
                                                        '23' in k or
                                                        '31' in k]
        for k in global_rel_pos_keys:
            h_check, w_check = net_dict[k].shape
            rel_pos_params = new_state_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            if h != h_check or w != w_check:
                rel_pos_params = F.interpolate(rel_pos_params, (h_check, w_check), mode='bilinear', align_corners=False)

            new_state_dict[k] = rel_pos_params[0, 0, ...]
    net_dict.update(new_state_dict)
    return net_dict