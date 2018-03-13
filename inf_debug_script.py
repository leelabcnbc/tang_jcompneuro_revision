"""this script gives inf or nan in pytorch 0.2.0
but error in 0.3.1"""

import torch
from torch.autograd import Variable

try:
    # raise RuntimeError('haha')
    x = Variable(torch.ones(2, 2), requires_grad=True)
    print(x)

    y = (x + 10) ** 100
    print(y)

    z = y * 5 + 3
    out = z.mean()

    print(z, out)

    out.backward()

    print(x.grad)
except RuntimeError as e:
    if e.args == ('value cannot be converted to type double without overflow: inf',):
        print('let us handle it!')
    else:
        print('we will not handle it')
        raise
print('done!')
