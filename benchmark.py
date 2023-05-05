import sys
import torch
import lpips
import argparse

sys.path.append("..")

from utils import load
from attacks import ATTACK_MAPPINGS
from dataloader import DATALOADER_MAPPINGS

from torch.utils.data import DataLoader, TensorDataset

def get_args_parser():
    parser = argparse.ArgumentParser('Benchmarking Semantic-X attacks', add_help=False)
    parser.add_argument('--root-dir', default="/scratch/itee/uqsswain/",
                        help="folder where all the artifacts lie; differs from system to system")
    parser.add_argument('--dataset-name', default="imagenet", type=str,
                        help="choose one of: imagenet, gtsrb, cifar10, cifar100, mnist")
    parser.add_argument('--attack-name', default="fgsm", type=str,
                        help="choose one of: fgsm, bim, pgd, cnw, deepfool, elastic")
    parser.add_argument('--benchmark-name', default="ppgd", type=str,
                        help="choose one of: ppgd, lp")
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    return parser

def main(args):
    loss_fn_alex = lpips.LPIPS(net='alex')
    attack_name = ATTACK_MAPPINGS.get(args.attack_name)

    # # loading x
    # train_dataloader = DATALOADER_MAPPINGS[args.dataset_name + "_x"](batch_size=1000, root=args.root_dir, dataset_len=1000)
    # xs = next(iter(train_dataloader))[0].to(args.device)

    # loading benchmark results
    benchmark_results = load(f"/scratch/itee/uqsswain/artifacts/spaa/objects/{args.dataset_name}/{args.benchmark_name}.pkl")
    x = benchmark_results["x"]
    benchmark_x_adv = benchmark_results[f"{args.benchmark_name}_x_adv"].detach().cpu()
    benchmark_lpips = loss_fn_alex(x, benchmark_x_adv)

    # loading spat-x results
    results = load(f"/scratch/itee/uqsswain/artifacts/spaa/objects/{args.dataset_name}/{args.attack_name}.pkl")
    x = results[attack_name.__name__]["x"]
    x_hat_adv = results[attack_name.__name__]["x_hat_adv"]
    modf_x_adv = results[attack_name.__name__]["modf_x_adv"]

    x_hat_adv = torch.Tensor(x_hat_adv)
    modf_x_adv = torch.Tensor(modf_x_adv)
    spat_lpips = loss_fn_alex(x, x_hat_adv)

    print(f"Benchmark LPIPS: {benchmark_lpips.mean()}")
    print(f"SPAT LPIPS: {spat_lpips.mean()}")

    dataset = TensorDataset(benchmark_x_adv)
    dataloader = DataLoader(dataset, batch_size=128)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Attack and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)


