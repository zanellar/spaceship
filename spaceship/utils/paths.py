import os
import spaceship

_PACKAGE_PATH = spaceship.__path__[0]  
  
PARAMS_PATH = os.path.join(_PACKAGE_PATH, os.pardir, "data", "params")  
RESULTS_PATH = os.path.join(_PACKAGE_PATH, os.pardir, "data", "results")
PLOTS_PATH = os.path.join(_PACKAGE_PATH, os.pardir, "data", "plots")
