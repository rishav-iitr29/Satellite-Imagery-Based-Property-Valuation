import torch
import pandas as pd
import numpy as np
import os
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from scipy.sparse import issparse
from tqdm import tqdm

# Edit file path accordingly

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = '../multimodal_fusion_model/model_output/best_multimodal_model.pth'
TEST_TABULAR_PATH = '../data/raw/test(1).csv'
IMAGE_DIR = '../data/images/test/images/'
OUTPUT_FILE = 'submission_predictions.csv'


class MultimodalDataset(Dataset):
    def __init__(self, X_tabular, property_ids, image_dir, y_labels=None, transform=None):
        self.X_tabular = torch.FloatTensor(X_tabular)
        self.property_ids = property_ids
        self.image_dir = image_dir
        self.transform = transform
        self.y_labels = torch.zeros(len(property_ids)) if y_labels is None else torch.FloatTensor(y_labels)

    def __len__(self):
        return len(self.property_ids)

    def __getitem__(self, idx):
        tabular_features = self.X_tabular[idx]
        label = self.y_labels[idx]
        property_id = self.property_ids[idx]

        image_path = os.path.join(self.image_dir, f"{property_id}.png")
        try:
            image = Image.open(image_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'tabular': tabular_features, 'label': label, 'id': property_id}


# MODEL ARCHITECTURE
class MultimodalModel(nn.Module):
    def __init__(self, tabular_input_dim):
        super(MultimodalModel, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        cnn_out_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.fusion = nn.Sequential(
            nn.Linear(cnn_out_dim + 128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, img, tab):
        img_feats = self.resnet(img)
        tab_feats = self.tabular_mlp(tab)
        combined = torch.cat((img_feats * 0.1, tab_feats), dim=1)
        return self.fusion(combined).squeeze(-1)


# LOAD DATA
if os.path.exists(f'{TEST_TABULAR_PATH}test.npy'):
    X_test = np.load(f'{TEST_TABULAR_PATH}test.npy', allow_pickle=True)
    test_ids = np.load(f'{TEST_TABULAR_PATH}test_ids.npy', allow_pickle=True)

    if X_test.shape == (): X_test = X_test.item()
    if issparse(X_test): X_test = X_test.toarray()
    X_test = np.array(X_test, dtype=np.float32)
else:
    raise FileNotFoundError("Input .npy files not found.")

# INITIALIZE MODEL
tabular_dim = X_test.shape[1]
model = MultimodalModel(tabular_input_dim=tabular_dim).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# DATALOADER
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = MultimodalDataset(X_test, test_ids, IMAGE_DIR, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

all_ids = []
all_preds = []

print(f"Starting inference on {DEVICE}...")
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting Prices", unit="batch"):
        imgs = batch['image'].to(DEVICE)
        tabs = batch['tabular'].to(DEVICE)
        ids = batch['id']

        outputs = model(imgs, tabs)

        all_ids.extend(ids)
        all_preds.extend(outputs.cpu().numpy())

# 6. SAVE
submission_df = pd.DataFrame({'id': all_ids, 'predicted_price': all_preds})
print(f"\nSuccessfully generated {len(submission_df)} predictions")

df = submission_df

# converting the prices back to dollar

df['id'] = df['id'].astype(str).str.extract('(\d+)').astype(int)

df['predicted_price'] = np.exp(df['predicted_price'])

df['predicted_price'] = df['predicted_price'].round().astype(int)

df.to_csv('final_predictions.csv', index=False)

print("Sample of formatted output:")
print(df.head())