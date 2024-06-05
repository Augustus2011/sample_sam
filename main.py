from utils.utils import setup
import numpy as np


if __name__ == "__main__":
    s = setup(3)
    s.print_math()
    s.set_seed()
    device=s.check_return_cuda()

    print(device)
    print(np.random.rand())