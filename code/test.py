import torch
from logsumexp_safe import logsumexp_new, logaddexp_new

print(torch.__version__)

if not hasattr(torch, 'logaddexp_old'):
    torch.logaddexp_old = torch.logaddexp  # save original def so we can call it above
    torch.logsumexp_old = torch.logsumexp  # save original def so we can call it above
torch.logaddexp = logaddexp_new
torch.Tensor.logaddexp = logaddexp_new
torch.logsumexp = logsumexp_new
torch.Tensor.logsumexp = logsumexp_new

if __name__ == "__main__":
    inf=float('inf') 

    # Some examples with logaddexp
    for a in -inf, 1., inf:
        for b in -inf, 2., inf:
            print("")
            for c in -inf, 3.:
                for safe_inf in False, True:
                    aa = torch.tensor(a, requires_grad=True)
                    bb = torch.tensor(b, requires_grad=True)
                    result = aa.logaddexp(bb, safe_inf=safe_inf).logaddexp(torch.tensor(c))
                    result.backward()
                    print(f"{'  safe' if safe_inf else 'unsafe'}: "
                          f"d=logaddexp({a}, {b}, {c})={result.item()}"
                          f"\t∂d/∂a={aa.grad.item()}\t∂d/∂b={bb.grad.item()}")
    
    # Some examples with tensorized logsumexp
    t = torch.tensor([[  2.,   3., -inf, -inf], 
                      [  5.,   7., -inf, -inf],
                      [-inf, -inf, -inf, -inf]], requires_grad=True)
    u = torch.tensor([[  1.,   0.,   1.,   0.],
                      [  1.,   0.,   1.,   0.],
                      [  1.,   0.,   1.,   0.]])
    
    for dim in 0, 1, (0,1):
        for keepdim in False, True:
            print(f"\ndim={dim}, keepdim={keepdim} -----")
            for safe_inf in False, True:
                x = t.clone()   # test that backward works when logsumexp is applied to a non-leaf
                y = x.logsumexp(dim=dim, keepdim=keepdim, safe_inf=safe_inf)
                z = u.sum(dim=dim, keepdim=keepdim)  # reduce size to match y
                (y*z).sum().backward()               # the product with z means that some elements of y's grad_output will be zero
                print(t.grad)
                t.grad.data.zero_()