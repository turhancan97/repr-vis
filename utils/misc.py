import torch


def convert_mae_to_vit(mae_state_dict):
    vit_state_dict = {}
    
    vit_state_dict['pos_embed'] = mae_state_dict['enc_pos_embed'].unsqueeze(0)  # Add batch dimension

    # add new dimension to the beginning of the tensor - Random dummy dimension for cls token
    vit_state_dict['pos_embed'] = torch.cat([torch.randn(1, 1, 768), vit_state_dict['pos_embed']], dim=1)
    
    # Copy patch embedding
    vit_state_dict['patch_embed.proj.weight'] = mae_state_dict['patch_embed.proj.weight']
    vit_state_dict['patch_embed.proj.bias'] = mae_state_dict['patch_embed.proj.bias']
    
    # Convert encoder blocks to standard blocks
    for i in range(12):  # Assuming 12 blocks
        # Rename encoder blocks to standard blocks
        vit_state_dict[f'blocks.{i}.norm1.weight'] = mae_state_dict[f'enc_blocks.{i}.norm1.weight']
        vit_state_dict[f'blocks.{i}.norm1.bias'] = mae_state_dict[f'enc_blocks.{i}.norm1.bias']
        
        vit_state_dict[f'blocks.{i}.attn.qkv.weight'] = mae_state_dict[f'enc_blocks.{i}.attn.qkv.weight']
        vit_state_dict[f'blocks.{i}.attn.qkv.bias'] = mae_state_dict[f'enc_blocks.{i}.attn.qkv.bias']
        
        vit_state_dict[f'blocks.{i}.attn.proj.weight'] = mae_state_dict[f'enc_blocks.{i}.attn.proj.weight']
        vit_state_dict[f'blocks.{i}.attn.proj.bias'] = mae_state_dict[f'enc_blocks.{i}.attn.proj.bias']
        
        vit_state_dict[f'blocks.{i}.norm2.weight'] = mae_state_dict[f'enc_blocks.{i}.norm2.weight']
        vit_state_dict[f'blocks.{i}.norm2.bias'] = mae_state_dict[f'enc_blocks.{i}.norm2.bias']
        
        vit_state_dict[f'blocks.{i}.mlp.fc1.weight'] = mae_state_dict[f'enc_blocks.{i}.mlp.fc1.weight']
        vit_state_dict[f'blocks.{i}.mlp.fc1.bias'] = mae_state_dict[f'enc_blocks.{i}.mlp.fc1.bias']
        
        vit_state_dict[f'blocks.{i}.mlp.fc2.weight'] = mae_state_dict[f'enc_blocks.{i}.mlp.fc2.weight']
        vit_state_dict[f'blocks.{i}.mlp.fc2.bias'] = mae_state_dict[f'enc_blocks.{i}.mlp.fc2.bias']
    
    # Convert final norm layer
    vit_state_dict['norm.weight'] = mae_state_dict['enc_norm.weight']
    vit_state_dict['norm.bias'] = mae_state_dict['enc_norm.bias']
    
    return vit_state_dict

def convert_maskfeat_to_vit(maskfeat_state_dict):
    vit_state_dict = {}
    
    # Copy cls token and pos embed directly
    vit_state_dict['cls_token'] = maskfeat_state_dict['cls_token']
    vit_state_dict['pos_embed'] = maskfeat_state_dict['pos_embed']
    
    # Copy patch embedding
    vit_state_dict['patch_embed.proj.weight'] = maskfeat_state_dict['patch_embed.proj.weight']
    vit_state_dict['patch_embed.proj.bias'] = maskfeat_state_dict['patch_embed.proj.bias']
    
    # Convert attention blocks
    for i in range(12):  # Assuming 12 blocks
        # Copy norm layers
        vit_state_dict[f'blocks.{i}.norm1.weight'] = maskfeat_state_dict[f'blocks.{i}.norm1.weight']
        vit_state_dict[f'blocks.{i}.norm1.bias'] = maskfeat_state_dict[f'blocks.{i}.norm1.bias']
        
        # Combine q, k, v into qkv
        q_weight = maskfeat_state_dict[f'blocks.{i}.attn.q.weight']
        k_weight = maskfeat_state_dict[f'blocks.{i}.attn.k.weight']
        v_weight = maskfeat_state_dict[f'blocks.{i}.attn.v.weight']
        vit_state_dict[f'blocks.{i}.attn.qkv.weight'] = torch.cat([q_weight, k_weight, v_weight], dim=0)
        
        q_bias = maskfeat_state_dict[f'blocks.{i}.attn.q.bias']
        k_bias = maskfeat_state_dict[f'blocks.{i}.attn.k.bias']
        v_bias = maskfeat_state_dict[f'blocks.{i}.attn.v.bias']
        vit_state_dict[f'blocks.{i}.attn.qkv.bias'] = torch.cat([q_bias, k_bias, v_bias], dim=0)
        
        # Copy projection layer
        vit_state_dict[f'blocks.{i}.attn.proj.weight'] = maskfeat_state_dict[f'blocks.{i}.attn.proj.weight']
        vit_state_dict[f'blocks.{i}.attn.proj.bias'] = maskfeat_state_dict[f'blocks.{i}.attn.proj.bias']
        
        # Copy MLP layers
        vit_state_dict[f'blocks.{i}.norm2.weight'] = maskfeat_state_dict[f'blocks.{i}.norm2.weight']
        vit_state_dict[f'blocks.{i}.norm2.bias'] = maskfeat_state_dict[f'blocks.{i}.norm2.bias']
        vit_state_dict[f'blocks.{i}.mlp.fc1.weight'] = maskfeat_state_dict[f'blocks.{i}.mlp.fc1.weight']
        vit_state_dict[f'blocks.{i}.mlp.fc1.bias'] = maskfeat_state_dict[f'blocks.{i}.mlp.fc1.bias']
        vit_state_dict[f'blocks.{i}.mlp.fc2.weight'] = maskfeat_state_dict[f'blocks.{i}.mlp.fc2.weight']
        vit_state_dict[f'blocks.{i}.mlp.fc2.bias'] = maskfeat_state_dict[f'blocks.{i}.mlp.fc2.bias']
    
    # Copy final norm layer
    vit_state_dict['norm.weight'] = maskfeat_state_dict['pred_hog_head.transforms.0.0.weight']
    vit_state_dict['norm.bias'] = maskfeat_state_dict['pred_hog_head.transforms.0.0.bias']
    
    return vit_state_dict