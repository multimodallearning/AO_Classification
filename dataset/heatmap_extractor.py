import torch
from pathlib import Path


class HeatmapExtractor:
    def __init__(self, resolution_HW: tuple[int, int], class2extract: int):
        self.H, self.W = resolution_HW
        self.class2extract = class2extract

        self.grid = torch.stack(torch.meshgrid(torch.arange(self.W), torch.arange(self.H), indexing='xy'), dim=-1)
        self.grid = self.grid.view(self.H * self.W, 2)  # (HW, 2)

    def extract_heatmap(self, yolo_file: Path):
        with open(yolo_file, 'r') as f:
            lines = f.readlines()

        heatmap = []
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.split())
            if int(class_id) == self.class2extract:
                x_center *= self.W
                y_center *= self.H
                width *= self.W
                height *= self.H

                centroid = torch.tensor([x_center, y_center]).unsqueeze(0)  # (1, 2)
                diff = self.grid - centroid  # (HW, 2)
                sigma = torch.tensor([width, height]).unsqueeze(0) / 2  # for more pointy gaussian in heatmap
                gaussian = 1 / (2 * torch.pi * sigma.prod()) * torch.exp(
                    -0.5 * torch.sum(diff.pow(2) / sigma.pow(2), dim=1))  # (HW,)
                gaussian /= gaussian.max()  # normalize to [0, 1]
                gaussian = gaussian.view(self.H, self.W)

                heatmap.append(gaussian)

        if len(heatmap) > 0:
            heatmap = torch.stack(heatmap, dim=0).mean(0)
            # normalize to [0, 1] which was lost in the mean operation
            heatmap /= heatmap.max()
        else:
            heatmap = torch.zeros(self.H, self.W)
        return heatmap


if __name__ == '__main__':
    from PIL import Image
    from matplotlib import pyplot as plt

    file_stem = "0003_0662359226_01_WRI-R1_M011"
    lbl_path = Path(f'/home/ron/Documents/AOClassification/data/yolo_labels/predictions/{file_stem}.txt')
    img = Image.open(f'data/img_only_front_all_left/{file_stem}.png').convert('L')

    extractor = HeatmapExtractor(resolution_HW=(img.height, img.width), class2extract=3)
    heatmap = extractor.extract_heatmap(lbl_path)
    heatmap = heatmap.flip(1)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(img, cmap='gray')
    axs[1].imshow(heatmap, cmap='hot')

    # surf plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y = torch.meshgrid(torch.arange(heatmap.shape[0]), torch.arange(heatmap.shape[1]))
    ax.plot_surface(X, Y, heatmap, cmap='hot')

    plt.show()
