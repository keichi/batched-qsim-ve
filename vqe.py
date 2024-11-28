#!/usr/bin/env python3

import numpy as np
import scipy
import subprocess


def main():
    n_param = 6

    def cost(phi):
        return float(subprocess.run(["build/qsim-vqe"] + [str(val) for val in phi], capture_output=True).stdout)

    cost_val = []

    def callback(phi):
        cost_val.append(cost(phi))

    init = np.random.rand(n_param)
    callback(init)
    res = scipy.optimize.minimize(cost, init,
                                  method="Powell",
                                  callback=callback,
                                  options={"maxiter": 100})

    print(res)
    print(cost_val)


if __name__ == "__main__":
    main()
