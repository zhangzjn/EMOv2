_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=None,
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
        sync_bn=False, out_indices=(1, 2, 3, 4), pretrained='../pretrained/emo2_5M.pth', frozen_stages=-1, norm_eval=False, ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[48, 72, 160, 288],
        in_index=[0, 1, 2, 3],
        channels=288,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),)

ratio = 1
bs_ratio = 4  # 0.00012 for 4 * 8

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00012 * ratio, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
#     dict(
#         type='PolyLR',
#         eta_min=0.0,
#         power=1.0,
#         begin=1500,
#         end=160000,
#         by_epoch=False,
#     )
# ]
max_iters = 80000 * 2
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1.0e-6, by_epoch=False, begin=0, end=1500),
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
