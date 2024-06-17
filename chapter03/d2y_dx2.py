import torch
import numpy

# 定义一个简单的函数 f(x) = x^3 - 2x^2 + x
def f(x):
    return x**3 - 2 * x**2 + x

# 创建一个需要梯度的张量
x = torch.tensor([1.0,2.], requires_grad=True)

# 计算一阶导数
y = f(x)
dy_dx = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=torch.ones(x.shape), create_graph=True)[0]

# 计算二阶导数
d2y_dx2 = torch.autograd.grad(outputs=dy_dx, inputs=x, grad_outputs=torch.ones(x.shape))[0]

# 打印结果
print(f"The first derivative of f at x={x} is {dy_dx}")
print(f"The second derivative of f at x={x} is {d2y_dx2}")