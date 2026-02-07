from utils.config import get_args
from main import fit

import warnings
warnings.filterwarnings("ignore")

args = get_args()

fit(args)