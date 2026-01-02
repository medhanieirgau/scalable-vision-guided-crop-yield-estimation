import torchvision.models as models
import torch.nn as nn
import torch
from torchvision.models import ResNet50_Weights, vit_b_16, ViT_B_16_Weights
'''
import sys
sys.path.append("/sailhome/mirgau/low-rank-satmae")
import models_vit_lora
import copy
from util.pos_embed import interpolate_pos_embed
from train_lora_util import activate_lora, deactivate_lora
from vit_lora_util import Block as LoraBlock
'''
def build_vit():
    # Load pretrained ViT
    vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

    # Access the existing head to get in_features
    in_features = vit.heads[0].in_features

    # Replace classification head with regression head
    vit.heads = nn.Linear(in_features, 1)
    return vit
'''
def build_vit(dino_ckpt_path="/atlas2/u/mirgau/lora_satmae/dinov2_vitb14_pretrain.pth"):
    model = models_vit_lora.__dict__['vit_base_patch16'](
        block_type = LoraBlock, patch_size=14, img_size=224, in_chans=3,
        num_classes=1, global_pool=False,
        use_ls=True
    )

    checkpoint = torch.load(dino_ckpt_path, map_location='cpu')

    if 'model' in checkpoint:
        checkpoint_model = checkpoint['model']
    elif 'teacher' in checkpoint:
        checkpoint_model = checkpoint['teacher']
        new_checkpoint_model = {}
        for k, v in checkpoint_model.items():
            new_checkpoint_model[k.replace('backbone.', '')] = copy.deepcopy(v)
        del checkpoint_model
        checkpoint_model = new_checkpoint_model
    else:
        checkpoint_model = checkpoint

    state_dict = model.state_dict()
    for k in ['patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias',
                'patch_embed.0.proj.weight', 'patch_embed.0.proj.bias', 'patch_embed.1.proj.weight',
                'patch_embed.1.proj.bias', 'patch_embed.2.proj.weight', 'patch_embed.2.proj.bias',
                'decoder_pred.0.weight', 'decoder_pred.0.bias', 'decoder_pred.1.weight', 'decoder_pred.1.bias',
                'decoder_pred.2.weight', 'decoder_pred.2.bias']:
        if k not in state_dict and k not in checkpoint_model:
            continue

        if k not in state_dict or (k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape):
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    interpolate_pos_embed(model, checkpoint_model)
    
    activate_lora(model, 'attn', 8)
    for name, param in model.named_parameters():
        if not ('lora' in name or 'head' in name or
                (False and 'patch_embed' in name)):
            param.requires_grad_(False)
    

    return model
'''

def build_resnet50():
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(2048, 1)
    return model

class ResNetWithZoneLatLon(nn.Module):
    def __init__(self, zone_vocab_size):
        super().__init__()
        
        # Load ResNet50 backbone without classification head
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # outputs (B, 2048, 1, 1)
        self.flatten = nn.Flatten()  # (B, 2048)

        # Zone embedding
        zone_embed_dim = 32 if zone_vocab_size > 100 else 16
        self.zone_embed = nn.Embedding(zone_vocab_size, zone_embed_dim)  # (B, zone_embed_dim)

        # Final regressor
        #in_dim = 2048 + zone_embed_dim + 2  # 2048 image + 16 zone + 2 coords
        in_dim = 2048 # 2048 image 
        self.regressor = nn.Linear(in_dim, 1)

    def forward(self, images):#, lat, lon, zone_ids):
        x = self.backbone(images)           # (B, 2048, 1, 1)
        x = self.flatten(x)                 # (B, 2048)

        #zone_vec = self.zone_embed(zone_ids)  # (B, zone_embed_dim)
        #coords = torch.cat([lat, lon], dim=1) # (B, 2)

        #full_input = torch.cat([x, zone_vec, coords], dim=1)  # (B, in_dim)
        #full_input = torch.cat([x, coords], dim=1)
        
        full_input = torch.cat([x], dim=1)  # (B, in_dim)
        return self.regressor(full_input)
