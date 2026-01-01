
import torch
import pytest
from diffmcmc.proposal.nets import TimeResNet, GaussianFourierProjection, ResNetBlock

def test_gaussian_fourier_projection():
    dim = 256
    scale = 10.0
    layer = GaussianFourierProjection(embed_dim=dim, scale=scale)
    
    t = torch.randn(10)
    out = layer(t)
    
    assert out.shape == (10, dim)
    assert not layer.W.requires_grad

def test_resnet_block_shape():
    batch_size = 5
    dim = 32
    time_dim = 64
    
    block = ResNetBlock(dim, time_dim)
    x = torch.randn(batch_size, dim)
    t_emb = torch.randn(batch_size, time_dim)
    
    out = block(x, t_emb)
    assert out.shape == x.shape

def test_time_resnet_forward():
    batch_size = 8
    dim = 16
    model = TimeResNet(dim)
    
    x = torch.randn(batch_size, dim)
    t = torch.rand(batch_size)
    
    out = model(x, t)
    assert out.shape == x.shape

def test_time_conditioning_sensitivity():
    # Ensure that changing t actually changes the output
    dim = 16
    model = TimeResNet(dim)
    x = torch.randn(1, dim)
    
    t1 = torch.tensor([0.1])
    t2 = torch.tensor([0.2])
    
    out1 = model(x, t1)
    out2 = model(x, t2)
    
    assert not torch.allclose(out1, out2), "Model output should depend on time t"

def test_gradient_flow():
    dim = 16
    model = TimeResNet(dim)
    x = torch.randn(5, dim, requires_grad=True)
    t = torch.rand(5)
    
    out = model(x, t)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None
    # Check parameters have grads
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} missing gradient"

if __name__ == "__main__":
    test_gaussian_fourier_projection()
    test_resnet_block_shape()
    test_time_resnet_forward()
    test_time_conditioning_sensitivity()
    test_gradient_flow()
    print("All Net tests passed!")
