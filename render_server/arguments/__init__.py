
from argparse import ArgumentParser, Namespace
import sys
import os
import shutil
import numpy as np
class GroupParams:
    pass
class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group
    
class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.asg_degree = 24
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cpu"
        self.prefetch_factor = 4
        self.num_workers = 16  
        self.memory_upper_limit = 20  # 10%
        self.eval = False
        self.eval_epoch_iteration = 5
        self.prune_method = '3dgs'

        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        #g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.debug_from = -1
        self.force_save_mem = False  # old config
        self.force_save_gpu_mem = False  # may reduce train speed
        self.dataset_shuffle = True  # train images shuffle
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.000016  
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.feature_sh_lr = 0.000125 
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.lambda_dssim = 0.2

        self.reset_lr_iterations = 0  

        self.percent_dense_split = 0.01  
        self.percent_dense_clone = 0.01 

        self.densify_and_prune_interval = 100

        self.densify_and_prune_from_iter = [500]
        self.densify_and_prune_until_iter = [15_000]

        self.densify_interval = 100
        self.densify_split_grad_threshold = 0.0002  
        self.densify_clone_grad_threshold = 0.0002  
        self.opacity_init_value = 0.1 
        self.scale_init = 1 
        self.opacity_reset_interval = 3000
        self.opacity_reset_value = 0.2  
        self.opacity_reset_only_ground = False

        self.prune_interval = 100
        self.prune_max_screen_size_interval = 3000
        self.prune_min_opacity = 0.005  
        self.prune_max_screen_size = 20

        self.recompute_poly_points_label_interval = 100

        self.img_max_res = 32000

        self.random_background = False

        # NTC
        self.use_ntc = False
        self.only_mlp = False
        self.ntc_lr = None
        self.ntc_conf_path = ""
        self.ntc_color_ratio = 0.1

        # ASG
        self.use_asg = False
        self.save_asg = False
        self.specular_lr_max_steps = 30_000

        # app
        self.appearance_optimization_opacity_from_iter = 0  

        # pgsr
        self.pgsr_single_view_weight = 0.015
        self.pgsr_single_view_weight_from_iter = 7000

        # self.pgsr_use_virtul_cam = False  # No use
        # self.pgsr_virtul_cam_prob = 0.5
        self.pgsr_multi_view_weight_from_iter = 7000
        self.pgsr_multi_view_ncc_weight = 0.15
        self.pgsr_multi_view_geo_weight = 0.03
        self.pgsr_multi_view_patch_size = 3
        self.pgsr_multi_view_sample_num = 102400
        self.pgsr_multi_view_pixel_noise_th = 1.0
        self.pgsr_multi_view_weight_avg_safe_ratio = 0.1  
        self.pgsr_multi_view_pixel_noise_safe_max = 3000
        self.pgsr_multi_view_abnormal_drop_from_iter = 12000

        # pgsr multi_view # for fly need more adj
        self.pgsr_ncc_scale = 1.0  
        self.pgsr_multi_view_num = 5 
        self.pgsr_multi_view_max_angle = 20  
        self.pgsr_multi_view_min_dis = 0.01  
        self.pgsr_multi_view_max_dis = 0.8  
        # pgsr mod skip

        self.pgsr_mod = 0  

      
        self.densify_abs_grad_threshold = 0.0008  
        self.abs_split_radii2D_threshold = 20

        self.random_background = False

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)