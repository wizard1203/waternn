import waternets

import torchvision.models as m


class MyModels:

	waternet = waternets.WaterNet()
	waternetsf = waternets.WaterNetSmallFC()
	waternetconvfc = waternets.WaterNetConvFC()