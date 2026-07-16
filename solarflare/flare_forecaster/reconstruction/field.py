import torch


def jacobian_function(B, x):

    J = []

    for i in range(3): #needs to be 3 because we have 3 components
        grad = torch.autograd.grad(
            B[:, i].sum(), x,
            create_graph = True, retain_graph = True


        )[0]
        J.append(grad)
    
    return torch.stack(J, dim = 1)

def curl(B, x):

    J = jacobian_function(B, x)
    return torch.stack([
        J[:, 2, 1] - J[:, 1, 2],
        J[:, 0, 2] - J[:, 2, 0],
        J[:, 1, 0] - J[:, 0, 1],


    ], dim = 1)

def divergence(B, x): 
    J = jacobian_function(B, x)

    return J[:, 0, 0] + J[:, 1, 1] + J[:, 2, 2]