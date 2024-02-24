import torch
import numpy as np
from voc_dataset import VOCDataset
from train_q2 import ResNet_TSNE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

def main():
    # Load I.N fine-tuned RN18 model
    model_path = '/content/hw1/q1_q2_classification/ResNet_TSNE2_checkpoint.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet_TSNE(len(VOCDataset.CLASS_NAMES))
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['ResNet_TSNE2_state_dict'])
    model.to(device)
    model.eval()

    # Load PASCAL VOC test set
    test_dataset = VOCDataset(split='test', size=(224, 224))

    # Randomly get 1000 images from test set
    indices = np.random.choice(len(test_dataset), 1000, replace=False)
    selected_data = Subset(test_dataset, indices)

    # Create a DataLoader for the selected data
    data_loader = DataLoader(selected_data, batch_size=32, shuffle=False)  # Batch size can be adjusted

    # Extracting features
    features = []
    labels = []
    with torch.no_grad():
        for img, target, _ in data_loader:
            img = img.to(device)
            output = model(img)
            features.append(output.cpu().numpy())

            # Assuming target is a multi-label one-hot encoded tensor,
            # we convert it to a list of class indices for each sample.
            # Here we assume the last dimension of target corresponds to class indices.
            labels.extend(target.cpu().numpy())

    # t-SNE projection
    features = np.vstack(features)
    tsne = TSNE(n_components=2, random_state=42)  # Set random state for reproducibility
    projection = tsne.fit_transform(features)

    # Define class colors and labels
    class_labels = VOCDataset.CLASS_NAMES
    colors = plt.cm.get_cmap('tab20', len(class_labels)).colors  # Using a built-in colormap with 20 distinct colors

    # Plot t-SNE in 2D
    plt.figure(figsize=(12, 8))
    
    # Iterate over all class labels
    for i, class_label in enumerate(class_labels):
        # For each class, find samples that are labeled with that class
        indices = [j for j, label in enumerate(labels) if label[i] == 1]
        # Scatter each class with its unique color and label
        plt.scatter(projection[indices, 0], projection[indices, 1], color=np.array(colors[i]), label=class_label, alpha=0.5)

    plt.title('2D t-SNE Projection of ResNet_TSNE Features')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend(loc='best')

    # Save and Show the plot
    plt.savefig('2d_tsne_plot.png', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
