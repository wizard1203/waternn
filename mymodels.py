import waternets

import torchvision.models as m

class MyModels:

    # waternet = waternets.WaterNet()
    # waternetsf = waternets.WaterNetSmallFC()
    # waternetconvfc = waternets.WaterNetConvFC()
    # waterdsnet = waternets.WaterDenseNet()
    # waterdsnetf = waternets.WaterDenseNetFinal()
    # waterdsnetf_in4_out58 = waternets.WaterDenseNet_in4_out58()
    # waterdsnetf_self_define = waternets.WaterDenseNet_self_define(growth_rate=opt.growth_rate, num_init_features=opt.num_init_features)

    def waterdsnetf_self_define(self, opt):
        return waternets.WaterDenseNet_self_define(growth_rate=opt.growth_rate, num_init_features=opt.num_init_features)

    def waterdsnetf(self, opt):
        return waternets.WaterDenseNetFinal()

    def waterdsnetf_in4_out58(self, opt):
        return waternets.WaterDenseNet_in4_out58()
