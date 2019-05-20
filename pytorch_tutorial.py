import torch
import numpy as np


# x = torch.ones(2, 2, requires_grad=True)
x = torch.tensor([1., 4., 2.])
x.requires_grad_(True)

# y = (3 * (x + 2) ** 1).mean()
y = (x * 2)
""" line below is not necessary if calculating grads of `y` respect to `x`. """
# y.requires_grad_(True)

""" when `y` is not a scalar, d\vec{y}/d\vec{x} will be a Jacobian matrix J, whose calculation is not 
supported by pytorch, instead you can calculate dy'/d\vec{x} = dy'/d\vec{y} * J = \vec{v}^T * J by 
passing a further variable `y'`. '"""
y.backward(torch.tensor([0.1, 1., 12.], dtype=torch.float))
# y.backward(torch.from_numpy(4. * np.ones([3,5], dtype=np.float32)))
# y.backward(torch.from_numpy(np.array([[1, 1, 1, 1]], dtype=np.float32)))
print(x.grad)

"""
why are `torch` and `torchvision` are designed to be separated?
"""