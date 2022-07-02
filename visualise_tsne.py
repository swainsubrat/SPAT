import torch
import pandas as pd
import plotly.express as px

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from sklearn.manifold import TSNE
from dataloader import load_mnist
from autoencoder import Encoder, Decoder

checkpoint = torch.load('./checkpoints/mnist_enc_dec.pth.tar')
print("Found Checkpoint :)")
encoder = checkpoint["encoder"]
decoder = checkpoint["decoder"]
encoder.to(device)
decoder.to(device)

_, _, test_dataloader = load_mnist(batch_size=1)
encoded_samples = []
for i, (image, label) in enumerate(test_dataloader):
    image = image.to(device)
    label = label.item()

    encoder.eval()

    with torch.no_grad():
        z = encoder(image)
    encoded_img = z.flatten().cpu().numpy()
    # print(z.shape, encoded_img.shape)
    encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_img)}
    encoded_sample['label'] = label
    encoded_samples.append(encoded_sample)

encoded_samples = pd.DataFrame(encoded_samples)
# print(encoded_samples)
tsne = TSNE(n_components=2)
tsne_results = tsne.fit_transform(encoded_samples.drop(['label'],axis=1))
fig = px.scatter(tsne_results, x=0, y=1,
                color=encoded_samples.label.astype(str),
                labels={'0': 'tsne-2d-one', '1': 'tsne-2d-two'})
fig.write_image("img/tsne_normal.png")
