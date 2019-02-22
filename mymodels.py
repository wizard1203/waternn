import waternets

import torchvision.models as m


class MyModels:

    # waternet = waternets.WaterNet()
    # waternetsf = waternets.WaterNetSmallFC()
    # waternetconvfc = waternets.WaterNetConvFC()
    # waterdsnet = waternets.WaterDenseNet()
    waterdsnetf = waternets.WaterDenseNetFinal()
    waterdsnetf_in4_out58 = waternets.WaterDenseNet_in4_out58()


