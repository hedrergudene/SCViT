from .info import __version__, HOMEPAGE, INFO
try:
    from .dataset import (PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST,
                                  BreastMNIST, BloodMNIST, TissueMNIST, OrganAMNIST, OrganCMNIST, OrganSMNIST,
                                  OrganMNIST3D, NoduleMNIST3D, AdrenalMNIST3D, FractureMNIST3D, VesselMNIST3D, SynapseMNIST3D)
    from .evaluator import Evaluator
except:
    print("Please install the required packages first. " +
          "Use `pip install -r requirements.txt`.")
