properties = {
    'nerve':{

        'weight_path': r'',

        'img_path': r'',
        'mask_path': r'',
        'atlas_path': r'',

        'data_file_path': r'',
    },
    'SAM_weight_path': r'',
    'model_size': {
            'small': {
                'encoder_embed_dim': 768,
                'encoder_depth': 12,
                'encoder_num_heads': 12,
                'encoder_global_attn_indexes': [2, 5, 8, 11],
                'name': 'sam_vit_b_01ec64.pth'
            },
            'medium': {
                'encoder_embed_dim': 1024,
                'encoder_depth': 24,
                'encoder_num_heads': 16,
                'encoder_global_attn_indexes': [5, 11, 17, 23],
                'name': 'sam_vit_l_0b3195.pth'
            },
            'large': {
                'encoder_embed_dim': 1280,
                'encoder_depth': 32,
                'encoder_num_heads': 16,
                'encoder_global_attn_indexes': [7, 15, 23, 31],
                'name': 'sam_vit_h_4b8939.pth'
            }
        }

}
