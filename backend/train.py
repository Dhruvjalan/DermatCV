import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

BODY_PARTS_DIR = "link to Body Parts Dataset"
FACIAL_SKIN_DIR = "link to Facial Skin Dataset"
CHECKPOINT_DIR = "checkpoints"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BodyPartClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(BodyPartClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 128), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class DiseaseClassifierWithGradCAM(nn.Module):
    def __init__(self, num_classes=8):
        super(DiseaseClassifierWithGradCAM, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 28 * 28, num_classes)
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        self.activations = x
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class NormalAutoencoder(nn.Module):
    def __init__(self):
        super(NormalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 14 * 14, 8)
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 32 * 14 * 14), nn.ReLU(),
            nn.Unflatten(1, (32, 14, 14)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )
    def forward(self, x):
        latent = self.encoder(x)
        normalized_latent = F.normalize(latent, p=2, dim=1)
        reconstructed = self.decoder(normalized_latent)
        return normalized_latent, reconstructed

class SkinPatchDataset(Dataset):
    def __init__(self, root_dir, transform=None, N=4):
        self.base_dataset = datasets.ImageFolder(root=root_dir, transform=transform)
        self.N = N

    def __len__(self):
        return len(self.base_dataset) * (self.N * self.N)

    def __getitem__(self, idx):
        base_img_idx = idx // (self.N * self.N)
        patch_pos_idx = idx % (self.N * self.N)
        image, _ = self.base_dataset[base_img_idx]
        C, H, W = image.shape
        patch_h, patch_w = H // self.N, W // self.N
        row = patch_pos_idx // self.N
        col = patch_pos_idx % self.N
        y1, y2 = row * patch_h, (row + 1) * patch_h
        x1, x2 = col * patch_w, (col + 1) * patch_w
        patch = image[:, y1:y2, x1:x2]
        return patch, patch

def run_training_pipeline(epochs=10, batch_size=16):
    print(f" usando dispositivo execution: {DEVICE}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    classifier_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    patch_base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    print("🔄 Loading Dataset folder paths from Google Drive...")
    body_dataset = datasets.ImageFolder(root=BODY_PARTS_DIR, transform=classifier_transforms)
    disease_dataset = datasets.ImageFolder(root=FACIAL_SKIN_DIR, transform=classifier_transforms)
    ae_patch_dataset = SkinPatchDataset(root_dir=FACIAL_SKIN_DIR, transform=patch_base_transform)

    body_loader = DataLoader(body_dataset, batch_size=batch_size, shuffle=True)
    disease_loader = DataLoader(disease_dataset, batch_size=batch_size, shuffle=True)
    ae_loader = DataLoader(ae_patch_dataset, batch_size=batch_size, shuffle=True)

    body_classes = body_dataset.classes
    disease_classes = disease_dataset.classes

    mapping_payload = {"body_classes": body_classes, "disease_classes": disease_classes}
    with open(os.path.join(CHECKPOINT_DIR, "class_mapping.json"), "w") as f:
        json.dump(mapping_payload, f, indent=4)
    print("📝 Written local 'class_mapping.json' verification map dependencies.")

    body_model = BodyPartClassifier(num_classes=len(body_classes)).to(DEVICE)
    disease_model = DiseaseClassifierWithGradCAM(num_classes=len(disease_classes)).to(DEVICE)
    ae_model = NormalAutoencoder().to(DEVICE)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()

    opt_body = optim.Adam(body_model.parameters(), lr=0.001)
    opt_disease = optim.Adam(disease_model.parameters(), lr=0.001)
    opt_ae = optim.Adam(ae_model.parameters(), lr=0.001)

    print("\n--- Training Phase 1: Body Part Classifier ---")
    body_model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in body_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            opt_body.zero_grad()
            loss = criterion_ce(body_model(images), labels)
            loss.backward()
            opt_body.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Body Part Loss: {total_loss/len(body_loader):.4f}")
    torch.save(body_model.state_dict(), os.path.join(CHECKPOINT_DIR, "body_classifier.pth"))

    print("\n--- Training Phase 2: Disease Classifier With Hooks ---")
    disease_model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in disease_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            opt_disease.zero_grad()
            loss = criterion_ce(disease_model(images), labels)
            loss.backward()
            opt_disease.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Disease Loss: {total_loss/len(disease_loader):.4f}")
    torch.save(disease_model.state_dict(), os.path.join(CHECKPOINT_DIR, "disease_classifier.pth"))

    print("\n--- Training Phase 3: Patch Autoencoder ---")
    ae_model.train()
    for epoch in range(epochs):
        total_loss = 0
        for patches, _ in ae_loader:
            patches = patches.to(DEVICE)
            opt_ae.zero_grad()
            _, reconstructed = ae_model(patches)
            loss = criterion_mse(reconstructed, patches)
            loss.backward()
            opt_ae.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Autoencoder Loss: {total_loss/len(ae_loader):.4f}")
    torch.save(ae_model.state_dict(), os.path.join(CHECKPOINT_DIR, "patch_autoencoder.pth"))

    print("\n💾 All weights successfully trained and saved into local './checkpoints/' folder.")

if __name__ == "__main__":
    run_training_pipeline(epochs=5, batch_size=16)