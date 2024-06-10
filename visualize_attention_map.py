import matplotlib.pyplot as plt
import numpy as np
import torch
import typing
import io
import os
import cv2

from PIL import Image
from torchvision import transforms
from model.model import ViT


if __name__=="__main__":
  # Load model
  chkpt = torch.load("/content/model.pt")

  model = ViT(n_classes=10, patch_size=4, hidden_dim=8, n_heads=2, n_blocks=2, chw=(3, 32, 32))
  model.load_state_dict(chkpt)
  model.eval()

  transform = transforms.Compose([
      transforms.Resize((32, 32)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
  ])

  im = Image.open("/content/Transformer/cat.png")
  x = transform(im)
  x.size()

  logits, att_mat = model(x.unsqueeze(0))

  att_mat = torch.stack(att_mat).squeeze(1)

  # Average the attention weights across all heads.
  att_mat = torch.mean(att_mat, dim=1)

  # To account for residual connections, we add an identity matrix to the
  # attention matrix and re-normalize the weights.
  residual_att = torch.eye(att_mat.size(1))
  aug_att_mat = att_mat + residual_att
  aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

  # Recursively multiply the weight matrices
  joint_attentions = torch.zeros(aug_att_mat.size())
  joint_attentions[0] = aug_att_mat[0]

  for n in range(1, aug_att_mat.size(0)):
      joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])
      
  # Attention from the output token to the input space.
  v = joint_attentions[-1]
  grid_size = int(np.sqrt(aug_att_mat.size(-1)))
  mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
  mask = cv2.resize(mask / mask.max(), im.size)[..., np.newaxis]
  result = (mask * im).astype("uint8")

  fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))

  ax1.set_title('Original')
  ax2.set_title('Attention Map')
  _ = ax1.imshow(im)
  _ = ax2.imshow(result)

  plt.savefig("result.png")



  