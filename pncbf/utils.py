def calculate_gradient(model, x):
    """
    Calculate the gradient of the model's output with respect to the input tensor.
    Args:
        model (torch.nn.Module): The model to calculate the gradient for.
        x (torch.Tensor): The input tensor.
    Returns:
        torch.Tensor: The gradient of the model's output with respect to the input tensor.
    """

    x.requires_grad_()

    model.zero_grad()
    output = model(x)
    output.sum().backward()

    return x.grad.cpu().detach().numpy()
