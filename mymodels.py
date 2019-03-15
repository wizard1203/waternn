import waternets

import torchvision.models as m
import config as opt

class MyModels:

    # waternet = waternets.WaterNet()
    # waternetsf = waternets.WaterNetSmallFC()
    # waternetconvfc = waternets.WaterNetConvFC()
    # waterdsnet = waternets.WaterDenseNet()
    waterdsnetf = waternets.WaterDenseNetFinal()
    waterdsnetf_in4_out58 = waternets.WaterDenseNet_in4_out58()
    waterdsnetf_self_define = waternets.WaterDenseNet_self_define(growth_rate=opt.growth_rate, num_init_features=opt.num_init_features)


