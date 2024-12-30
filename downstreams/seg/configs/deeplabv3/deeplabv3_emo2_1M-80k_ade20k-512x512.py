_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='EMO2',
        dim_in=3,
        num_classes=80,
        depths=(2, 2, 8, 3),
        embed_dims=(32, 48, 80, 180),
        exp_ratios=(2., 2.5, 3.0, 3.5),
        norm_layers=('bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'),
        act_layers=('silu', 'silu', 'gelu', 'gelu'),
        dw_kss=[5, 5, 5, 5], dim_heads=[16, 16, 20, 20], window_sizes=[10, 10, 10, 10],
        hybrid_eopss=[[0], [0], [3], [3]],
        conv_kss=[1, 1, 1, 1], conv_groupss=[1, 1, 1, 1],
        qkv_bias=True, attn_drop=0., drop=0., drop_path=0.04036,
        v_group=False, attn_pre=False, ls_value=1e-6,
        sync_bn=False, out_indices=(1, 2, 3, 4), pretrained='../pretrained/emo2_1M.pth', frozen_stages=-1, norm_eval=False,),
    decode_head=dict(in_channels=180, channels=256, num_classes=150),
    auxiliary_head=dict(in_channels=80, num_classes=150)
)

ratio = 1
bs_ratio = 4  # 0.00012 for 4 * 8

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(_delete_=True, type='AdamW', lr=0.00012 * ratio, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                   'relative_position_bias_table': dict(decay_mult=0.),
                                                   'norm': dict(decay_mult=0.)}),
    clip_grad=dict(_delete_=True, max_norm=0.1, norm_type=2), )

# learning rate
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
#     dict(
#         type='CosineAnnealingLR',
#         begin=0,
#         # T_max=max_epochs,
#         end=80000,
#         by_epoch=False,
#         eta_min=0)
# ]
max_iters = 80000
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        begin=max_iters // 2,
        T_max=max_iters // 2,
        end=max_iters,
        by_epoch=False,
        eta_min=0)
]

train_dataloader = dict(
    batch_size=2 * bs_ratio * ratio,
    num_workers=min(2 * bs_ratio * ratio, 8),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,)
test_dataloader = val_dataloader

