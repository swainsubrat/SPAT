import lpips
import torch

def calculate_lpips(args, result, attack_name, x, linf=True):
    # LPIPS
    if not args.skip_orig:
        x_adv = result[attack_name.__name__]["x_adv"]

    x_hat_adv  = result[attack_name.__name__]["x_hat_adv"]
    modf_x_adv = result[attack_name.__name__]["modf_x_adv"]

    loss_fn_alex = lpips.LPIPS(net='alex').to(args.device) # best forward scores

    img_modf = torch.Tensor(x_hat_adv).to(args.device)
    img = x.detach().cpu().to(args.device)

    orig_lpips = None
    orig_linf, modf_linf = None, None
    if not args.skip_orig:
        img_orig = torch.Tensor(x_adv).to(args.device) # image should be RGB, IMPORTANT: normalized to [-1,1]
        orig_lpips = loss_fn_alex(img, img_orig)
        if linf:
            orig_linf = torch.max(torch.abs(img - img_orig))
    if linf:
        modf_linf = torch.max(torch.abs(img - img_modf))
    modf_lpips = loss_fn_alex(img, img_modf)

    return orig_lpips, modf_lpips, orig_linf, modf_linf
