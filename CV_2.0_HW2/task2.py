import torch
import triton
import triton.language as tl

def layernorm_forward_kernel(
    x_ptr, weight_ptr, bias_ptr, output_ptr,
    mean_ptr, rstd_ptr,
    M, N, eps,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offset = pid * N
    
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    
    x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
    
    sum_x = tl.sum(x, axis=0)
    mean = sum_x / N
    tl.store(mean_ptr + pid, mean)
    
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(rstd_ptr + pid, rstd)
    
    x_hat = x_centered * rstd
    weight = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    output = x_hat * weight + bias
    
    tl.store(output_ptr + row_offset + cols, output, mask=mask)


# Компилируем и добавляем autotune в старом стиле
layernorm_forward_kernel_jitted = triton.jit(layernorm_forward_kernel)
layernorm_forward_kernel_autotuned = triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_N': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 1024}, num_warps=8),
    ],
    key=['N'],
)(layernorm_forward_kernel_jitted)


def layernorm_forward_triton(x, weight, bias, eps=1e-5):
    M, N = x.shape
    output = torch.empty_like(x)
    mean = torch.empty(M, device=x.device)
    rstd = torch.empty(M, device=x.device)
    
    grid = (M,)
    layernorm_forward_kernel_autotuned[grid](x, weight, bias, output, mean, rstd, M, N, eps)
    
    return output, mean, rstd


# ============ BACKWARD PASS ============

def layernorm_backward_kernel(
    x_ptr, dy_ptr, weight_ptr,
    dx_ptr, dweight_ptr, dbias_ptr,
    mean_ptr, rstd_ptr,
    M, N,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row_offset = pid * N
    
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    
    x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
    dy = tl.load(dy_ptr + row_offset + cols, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    mean = tl.load(mean_ptr + pid)
    rstd = tl.load(rstd_ptr + pid)
    
    x_hat = (x - mean) * rstd
    dy_weighted = dy * weight
    
    sum_dy_weighted = tl.sum(dy_weighted, axis=0)
    sum_dy_weighted_x_hat = tl.sum(dy_weighted * x_hat, axis=0)
    
    dx = rstd / N * (N * dy_weighted - sum_dy_weighted - x_hat * sum_dy_weighted_x_hat)
    tl.store(dx_ptr + row_offset + cols, dx, mask=mask)
    
    dweight_val = tl.sum(dy_weighted * x_hat, axis=0)
    dbias_val = tl.sum(dy, axis=0)
    
    for i in range(N):
        if i < N:
            tl.atomic_add(dweight_ptr + i, dweight_val)
            tl.atomic_add(dbias_ptr + i, dbias_val)


layernorm_backward_kernel_jitted = triton.jit(layernorm_backward_kernel)


def layernorm_backward_triton(x, dy, weight, mean, rstd):
    M, N = x.shape
    dx = torch.empty_like(x)
    dweight = torch.zeros_like(weight)
    dbias = torch.zeros_like(weight)
    
    grid = (M,)
    layernorm_backward_kernel_jitted[grid](x, dy, weight, dx, dweight, dbias, mean, rstd, M, N)
    
    return dx, dweight, dbias


# ============ AUTODIFF FUNCTION ============

class TritonLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps=1e-5):
        output, mean, rstd = layernorm_forward_triton(x, weight, bias, eps)
        ctx.save_for_backward(x, weight, mean, rstd)
        ctx.eps = eps
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, mean, rstd = ctx.saved_tensors
        dx, dweight, dbias = layernorm_backward_triton(x, grad_output, weight, mean, rstd)
        return dx, dweight, dbias, None


def layernorm_triton(x, weight, bias, eps=1e-5):
    return TritonLayerNorm.apply(x, weight, bias, eps)


# ============ ЭТАЛОН ДЛЯ СРАВНЕНИЯ ============

def layernorm_torch(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, unbiased=False, keepdim=True)
    rstd = 1.0 / torch.sqrt(var + eps)
    x_hat = (x - mean) * rstd
    output = x_hat * weight + bias
    return output


# ============ ПРОВЕРКА ============

def test_correctness():
    torch.manual_seed(42)
    M, N = 1024, 768
    x = torch.randn(M, N, device='cuda', requires_grad=True)
    weight = torch.randn(N, device='cuda', requires_grad=True)
    bias = torch.randn(N, device='cuda', requires_grad=True)
    eps = 1e-5
    
    out_torch = layernorm_torch(x, weight, bias, eps)
    out_triton = layernorm_triton(x, weight, bias, eps)
    torch.testing.assert_close(out_torch, out_triton, rtol=1e-3, atol=1e-3)
    print("✓ Forward pass passed")
    
    out_torch.sum().backward()
    grad_torch = (x.grad.clone(), weight.grad.clone(), bias.grad.clone())
    
    x.grad, weight.grad, bias.grad = None, None, None
    out_triton.sum().backward()
    grad_triton = (x.grad, weight.grad, bias.grad)
    
    torch.testing.assert_close(grad_torch[0], grad_triton[0], rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(grad_torch[1], grad_triton[1], rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(grad_torch[2], grad_triton[2], rtol=1e-3, atol=1e-3)
    print("✓ Backward pass passed")


# ============ БЕНЧМАРК ============

def benchmark():
    shapes = [(128, 256), (512, 512), (1024, 768), (4096, 1024)]
    
    print("\n" + "="*60)
    print(f"{'Shape':<15} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10}")
    print("="*60)
    
    for M, N in shapes:
        x = torch.randn(M, N, device='cuda')
        weight = torch.randn(N, device='cuda')
        bias = torch.randn(N, device='cuda')
        
        for _ in range(10):
            layernorm_torch(x, weight, bias)
            layernorm_triton(x, weight, bias)
        
        torch.cuda.synchronize()
        
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        
        starter.record()
        for _ in range(100):
            _ = layernorm_torch(x, weight, bias)
        ender.record()
        torch.cuda.synchronize()
        torch_time = starter.elapsed_time(ender) / 100
        
        starter.record()
        for _ in range(100):
            _ = layernorm_triton(x, weight, bias)
        ender.record()
        torch.cuda.synchronize()
        triton_time = starter.elapsed_time(ender) / 100
        
        print(f"{M}x{N:<10} {torch_time:<15.3f} {triton_time:<15.3f} {torch_time/triton_time:<10.2f}x")


if __name__ == "__main__":
    test_correctness()
    benchmark()