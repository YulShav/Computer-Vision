import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def layernorm_forward_kernel(
    x_ptr, w_ptr, bias_ptr, res_ptr, mean_ptr, rstd_ptr,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    row_offset = row * N
    
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
    
    sum_x = tl.sum(x, axis=0)
    mean = sum_x / N
    tl.store(mean_ptr + row, mean)
    
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    tl.store(rstd_ptr + row, rstd)
    
    x_hat = x_centered * rstd
    w = tl.load(w_ptr + cols, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    output = x_hat * w + bias
    
    tl.store(res_ptr + row_offset + cols, output, mask=mask)


@triton.jit
def layernorm_backward_kernel(
    x_ptr, dy_ptr, w_ptr,
    dx_ptr, dw_ptr, dbias_ptr,
    mean_ptr, rstd_ptr, N,

    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0)
    row_offset = row * N
    
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
    dy = tl.load(dy_ptr + row_offset + cols, mask=mask, other=0.0)
    w = tl.load(w_ptr + cols, mask=mask, other=1.0)
    mean = tl.load(mean_ptr + row)
    rstd = tl.load(rstd_ptr + row)
    
    x_hat = (x - mean) * rstd
    dy_weighted = dy * w
    
    sum_dy_weighted = tl.sum(dy_weighted, axis=0)
    sum_dy_weighted_x_hat = tl.sum(dy_weighted * x_hat, axis=0)
    
    dx = rstd / N * (N * dy_weighted - sum_dy_weighted - x_hat * sum_dy_weighted_x_hat)
    tl.store(dx_ptr + row_offset + cols, dx, mask=mask)
    
    dw_val = tl.sum(dy_weighted * x_hat, axis=0)
    dbias_val = tl.sum(dy, axis=0)
    
    for i in range(N):
        if i < N:
            tl.atomic_add(dw_ptr + i, dw_val)
            tl.atomic_add(dbias_ptr + i, dbias_val)


def layernorm_forward_triton(x, w, bias, eps=1e-5):
    M, N = x.shape
    output = torch.empty_like(x)
    mean = torch.empty(M, device=x.device)
    rstd = torch.empty(M, device=x.device)
    
    grid = (M,)
    layernorm_forward_kernel[grid](
        x, w, bias, 
        output, mean, rstd,
        N, eps
    )
    
    return output, mean, rstd

def layernorm_backward_triton(x, dy, w, mean, rstd):
    M, N = x.shape
    dx = torch.empty_like(x)
    dw = torch.zeros_like(w)
    dbias = torch.zeros_like(w)
    
    grid = (M,)
    layernorm_backward_kernel[grid](
        x, dy, w, 
        dx, dw, dbias, 
        mean, rstd, N
    )
    
    return dx, dw, dbias


class TritonLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, bias, eps=1e-5):
        output, mean, rstd = layernorm_forward_triton(x, w, bias, eps)
        ctx.save_for_backward(x, w, mean, rstd)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, w, mean, rstd = ctx.saved_tensors
        dx, dw, dbias = layernorm_backward_triton(x, grad_output, w, mean, rstd)
        return dx, dw, dbias, None


def layernorm_triton(x, w, bias, eps=1e-5):
    return TritonLayerNorm.apply(x, w, bias, eps)


def layernorm_torch(x, w, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + eps)
    x_hat = (x - mean) * rstd
    output = x_hat * w + bias
    return output


def test_correctness():
    torch.manual_seed(42)
    M, N = 1024, 768
    x = torch.randn(M, N, device='cuda', requires_grad=True)
    w = torch.randn(N, device='cuda', requires_grad=True)
    bias = torch.randn(N, device='cuda', requires_grad=True)
    eps = 1e-5
    
    out_torch = layernorm_torch(x, w, bias, eps)
    out_triton = layernorm_triton(x, w, bias, eps)
    torch.testing.assert_close(out_torch, out_triton, rtol=1e-3, atol=1e-3)
    print("Forward pass passed")
    
    out_torch.sum().backward()
    grad_torch_x = x.grad.clone()
    grad_torch_w = w.grad.clone()
    grad_torch_bias = bias.grad.clone()
    
    x.grad = None
    w.grad = None
    bias.grad = None
    
    out_triton.sum().backward()
    
    torch.testing.assert_close(x.grad, grad_torch_x, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(w
                               .grad, grad_torch_w, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(bias.grad, grad_torch_bias, rtol=1e-3, atol=1e-3)
    print("Backward pass passed")


def benchmark():
    shapes = [(128, 256), (512, 512), (1024, 768), (4096, 1024)]
    
    print("\n" + "="*60)
    print(f"{'Shape':<15} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10}")
    print("="*60)
    
    for M, N in shapes:
        x = torch.randn(M, N, device='cuda')
        w = torch.randn(N, device='cuda')
        bias = torch.randn(N, device='cuda')
        
        for _ in range(10):
            layernorm_torch(x, w, bias)
            layernorm_triton(x, w, bias)
        
        torch.cuda.synchronize()
        
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        
        starter.record()
        for _ in range(100):
            _ = layernorm_torch(x, w, bias)
        ender.record()
        torch.cuda.synchronize()
        torch_time = starter.elapsed_time(ender) / 100
        
        starter.record()
        for _ in range(100):
            _ = layernorm_triton(x, w, bias)
        ender.record()
        torch.cuda.synchronize()
        triton_time = starter.elapsed_time(ender) / 100
        
        print(f"{M}x{N:<10} {torch_time:<15.3f} {triton_time:<15.3f} {torch_time/triton_time:<10.2f}x")


if __name__ == "__main__":
    test_correctness()
    benchmark()