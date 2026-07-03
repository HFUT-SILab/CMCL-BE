from easydict import EasyDict as edict

config = edict()
# config.dataset = "emoreIresNet" # training dataset
config.dataset = "polyu_data" # training dataset
config.embedding_size = 512 # embedding size of model
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 32 # batch size per GPU
config.lr = 0.1
config.output = "output/30epoch_sty4000_data" # train model output folder
config.global_step=0 # step to resume
config.save_epoch=1
config.s=64.0
config.m=0.50
config.std=0.05
config.num_epoch = 35

config.loss = "AdaFace"  #  Option : ElasticArcFace, ArcFace, ElasticCosFace, CosFace, MLLoss, ElasticArcFacePlus, ElasticCosFacePlus, AdaFace

if (config.loss=="ElasticArcFacePlus"):
    config.s = 64.0
    config.m = 0.50
    config.std = 0.0175
elif (config.loss=="ElasticArcFace"):
    config.s = 64.0
    config.m = 0.50
    config.std = 0.05
if (config.loss=="ElasticCosFacePlus"):
    config.s = 64.0
    config.m = 0.35
    config.std = 0.02
elif (config.loss=="ElasticCosFace"):
    config.s = 64.0
    config.m = 0.35
    config.std = 0.05
if (config.loss=="AdaFace"):
    config.s = 64.0
    config.m = 0.5
    config.h = 0.333
    config.t_alpha = 0.01

    config.num_epoch = 30

# training efficiency / early-stop settings
config.max_train_epoch = 30
config.dynamic_save_start_epoch = 16
config.early_stop_min_epoch = 30
config.early_stop_patience = 2
config.early_stop_extra_epochs = 2
config.early_stop_min_delta = 5e-5

# type of network to train [iresnet100 | iresnet50]
config.network = "iresnet50"  # "mobilefacenet_qat.MobileFaceNet"   #"iresnet50"  # "iresnet100"
config.SE=False # SEModule
config.warmup_epoch = -1
config.val_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
config.eval_step= 958 #33350
def lr_step_func(epoch):
    return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
        [m for m in [22, 30, 40] if m - 1 <= epoch])
config.lr_func = lr_step_func

# if config.dataset == "all_data":
#     config.rec = r'G:\dataset\vein_opendata\all_aug\train'
#     config.num_classes = 1502
#     config.num_image = 25248
#     # 748 12552
#     # config.num_epoch = 30   #  [22, 30, 35]
#     config.warmup_epoch = -1
#     config.val_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
#     config.eval_step= 958 #33350
#     def lr_step_func(epoch):
#         return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
#             [m for m in [22, 30, 40] if m - 1 <= epoch])
#     config.lr_func = lr_step_func

# elif config.dataset == "casia_data":
#     config.rec = r'G:\dataset\vein_opendata\casia_aug\train'
#     config.num_classes = 200
#     config.num_image = 2400
#     # 748 12552
#     # config.num_epoch = 30   #  [22, 30, 35]
#     config.warmup_epoch = -1
#     config.val_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
#     config.eval_step= 958 #33350
#     def lr_step_func(epoch):
#         return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
#             [m for m in [22, 30, 40] if m - 1 <= epoch])
#     config.lr_func = lr_step_func

# elif config.dataset == "hfut_data":
#     config.rec = r'G:\dataset\vein_opendata\hfut_aug\train'
#     config.num_classes = 202
#     config.num_image = 4848
#     # 748 12552
#     # config.num_epoch = 30   #  [22, 30, 35]
#     config.warmup_epoch = -1
#     config.val_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
#     config.eval_step= 958 #33350
#     def lr_step_func(epoch):
#         return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
#             [m for m in [22, 30, 40] if m - 1 <= epoch])
#     config.lr_func = lr_step_func

# elif config.dataset == "polyu_data":
#     config.rec = r'G:\dataset\vein_opendata\polyu_aug\train'
#     config.num_classes = 500
#     config.num_image = 6000
#     # 748 12552
#     # config.num_epoch = 30   #  [22, 30, 35]
#     config.warmup_epoch = -1
#     config.val_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
#     config.eval_step= 958 #33350
#     def lr_step_func(epoch):
#         return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
#             [m for m in [22, 30, 40] if m - 1 <= epoch])
#     config.lr_func = lr_step_func

# elif config.dataset == "tongji_data":
#     config.rec = r'G:\dataset\vein_opendata\tongji_aug\train'
#     config.num_classes = 600
#     config.num_image = 12000
#     # 748 12552
#     # config.num_epoch = 30   #  [22, 30, 35]
#     config.warmup_epoch = -1
#     config.val_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
#     config.eval_step= 958 #33350
#     def lr_step_func(epoch):
#         return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
#             [m for m in [22, 30, 40] if m - 1 <= epoch])
#     config.lr_func = lr_step_func

# elif config.dataset == "syn_data":
#     config.rec = r'G:\dataset\vein_opendata\syn_aug'
#     config.num_classes = 4000
#     config.num_image = 80000
#     # 748 12552
#     # config.num_epoch = 30   #  [22, 30, 35]
#     config.warmup_epoch = -1
#     config.val_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
#     config.eval_step= 958 #33350
#     def lr_step_func(epoch):
#         return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
#             [m for m in [22, 30, 40] if m - 1 <= epoch])
#     config.lr_func = lr_step_func
