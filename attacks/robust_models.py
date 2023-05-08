from robustbench.utils import load_model

# models = ["Rebuffi2021Fixing_70_16_cutmix_extra", "Gowal2021Improving_70_16_ddpm_100m",
#           "Rebuffi2021Fixing_70_16_cutmix_ddpm", "Sehwag2021Proxy_ResNest152"]

model = load_model(
    model_name='Carmon2019Unlabeled',
    dataset='cifar10',
    model_dir='/scratch/itee/uqsswain/artifacts/spaa/robust_models/cifar10',
    threat_model='Linf')

