import pandas as pd
import plotly.express as px
import torch

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")

from sklearn.manifold import TSNE

from dataloader import load_cifar
from models.autoencoder import CIFAR10Autoencoder

model = CIFAR10Autoencoder.load_from_checkpoint("./lightning_logs/version_36/checkpoints/checkpoint.ckpt")
model.eval()

_, _, test_dataloader = load_cifar(batch_size=100)
images = next(iter(test_dataloader))
print("Loaded cifar 10 dataloader")

encoded_samples = []
image = image.to(device)
label = label.item()

z = model.get_z(image)
encoded_img = z.flatten().cpu().numpy()
    # print(z.shape, encoded_img.shape)
encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_img)}
print(encoded_sample)

# encoded_sample['label'] = label
# encoded_samples.append(encoded_sample)

# # for i, (image, label) in enumerate(test_dataloader):
# #     image = image.to(device)
# #     label = label.item()

# #     z = model.get_z(image)
# #     encoded_img = z.flatten().cpu().numpy()
# #     # print(z.shape, encoded_img.shape)
# #     encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_img)}
# #     encoded_sample['label'] = label
# #     encoded_samplxes.append(encoded_sample)

# #     if i == 100:
# #         break

# encoded_samples = pd.DataFrame(encoded_samples)
# # print(encoded_samples)
# tsne = TSNE(n_components=2)
# tsne_results = tsne.fit_transform(encoded_samples.drop(['label'],axis=1))
# fig = px.scatter(tsne_results, x=0, y=1,
#                     color=encoded_samples.label.astype(str),
#                     color_discrete_map={"0":"red", "1":"blue", "2":"yellow", "3":"gray", "4":"brown", "5":"aqua", "6":"maroon", "7":"purple", "8":"teal", "9":"lime"},
#                     labels={'0': 'dimension-1', '1': 'dimension-2'})
# fig.write_image("img/tsne_new_normal.png")
if __name__ == "__main__":
    print("In main")
    torch.multiprocessing.freeze_support()