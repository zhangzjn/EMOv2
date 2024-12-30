import model.lib_teacher.tresnet_v2
import model.lib_emo.emo2
import model.lib_emo.emo2_ablation
import model.lib_emo.emo2_columnar
import model.lib_emo.emo2_HR

from timm.models._registry import _model_entrypoints
from . import MODEL


for timm_name, timm_fn in _model_entrypoints.items():
	MODEL.register_module(timm_fn, f'timm_{timm_name}')


if __name__ == '__main__':
	print()
	