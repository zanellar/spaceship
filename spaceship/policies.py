import numpy as np

def void_policy(t_now):
    return dict(
        f1 = np.array([0,0,0]),        # TODO here we can define the policy as a function of time
        tau1 = np.array([0,0,0]),
    )


def policy1(t_now):
    raise NotImplementedError 