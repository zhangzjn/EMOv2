_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
    './retinanet_tta.py'
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='EMO2',
        dim_in=3,
        num_classes=80,
        depths=(3, 3, 9, 3),
        embed_dims=(48, 72, 160, 288),
        exp_ratios=(2., 3., 4., 4.),
        norm_layers=('bn_2d', 'bn_2d', 'ln_2d', 'ln_2d'),
        act_layers=('silu', 'silu', 'gelu', 'gelu'),
        dw_kss=[5, 5, 5, 5], dim_heads=[16, 24, 32, 32], window_sizes=[10, 10, 10, 10],
        hybrid_eopss=[[0], [0], [3], [3]],
        conv_kss=[1, 1, 1, 1], conv_groupss=[1, 1, 1, 1],
        qkv_bias=True, attn_drop=0., drop=0., drop_path=0.05,
        v_group=False, attn_pre=False, ls_value=1e-6,
        sync_bn=False, out_indices=(1, 2, 3, 4), pretrained='../pretrained/emo2_5M.pth', frozen_stages=1, norm_eval=True,),
    neck=dict(in_channels=[48, 72, 160, 288],)
)

ratio = 1
bs_ratio = 2  # 0.0002 for 2 * 8

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(_delete_=True, type='AdamW', lr=0.0002 * ratio, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                   'relative_position_bias_table': dict(decay_mult=0.),
                                                   'norm': dict(decay_mult=0.)}),
    clip_grad=dict(max_norm=0.1, norm_type=2), )

# learning rate
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
#     dict(
#         type='CosineAnnealingLR',
#         begin=0,
#         # T_max=max_epochs,
#         end=12,
#         by_epoch=True,
#         eta_min=0)
# ]
max_epochs = 12
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        begin=max_epochs // 2,
        T_max=max_epochs // 2,
        end=max_epochs,
        by_epoch=True,
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
