import matplotlib.pyplot as plt

from dataset.grazpedwri_dataset import GrazPedWriDataset

dataset = GrazPedWriDataset('val', fold=1, use_yolo_predictions=False)
data = dataset[0]
print(data['file_name'])
print(dataset.CLASS_LABELS[data['y'].argmax()])
print(data['report'])

fig_size = (5, 10)
save_dir = '/home/ron/Documents/Konferenzen/BVM 2025/plots'

plt.figure(figsize=fig_size)
plt.imshow(data['image'].squeeze().numpy(), cmap='gray')
plt.axis('off')
plt.savefig(save_dir + '/img.png', bbox_inches='tight', pad_inches=0)

plt.figure(figsize=fig_size)
plt.imshow(data['image'].squeeze().numpy(), cmap='gray')
plt.imshow(data['segmentation'].float().argmax(0), alpha=data['segmentation'].any(0).float() * .8, cmap='tab20',
              interpolation='nearest')
plt.axis('off')
plt.savefig(save_dir + '/mul_seg.png', bbox_inches='tight', pad_inches=0)

plt.figure(figsize=fig_size)
plt.imshow(data['image'].squeeze().numpy(), cmap='gray')
plt.imshow(data['segmentation'].any(0), alpha=data['segmentation'].any(0).float() * .8, cmap='tab20',
              interpolation='nearest')
plt.axis('off')
plt.savefig(save_dir + '/bin_seg.png', bbox_inches='tight', pad_inches=0)

plt.figure(figsize=fig_size)
plt.imshow(data['image'].squeeze().numpy(), cmap='gray')
plt.imshow(data['fracture_heatmap'].squeeze(0), cmap='hot', alpha=.8)
plt.axis('off')
plt.savefig(save_dir + '/frac_loc.png', bbox_inches='tight', pad_inches=0)

plt.show()
