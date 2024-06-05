import math
import torch
import numpy as np

class setup:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def print_math(self) -> None:
        print(math.pi)

    def check_return_cuda(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
