import os
import torch
from tqdm import tqdm
import pandas as pd

from dataset import create_df, BCSSTestDataset
from model import AttentionUNet
from torchvision import transforms as T
from PIL import Image
import numpy as np

# Allow user to specify GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "5"  # Change "0" to the desired GPU index


def predict_image(model, image, device):
    model.eval()
    model.to(device)

    # ensure image is a tensor: accept PIL Image, numpy array, or torch tensor
    if isinstance(image, Image.Image):
        t = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = t(image)
    elif isinstance(image, np.ndarray):
        # assume HWC RGB uint8
        image = Image.fromarray(image)
        t = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        image = t(image)
    elif not isinstance(image, torch.Tensor):
        raise TypeError(f"Unsupported image type: {type(image)}")

    image = image.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        output = model(image)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked


def main():
    # paths (same as notebook)
    TEST_IMAGE_PATH = './BCSS/test/'

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create dataframe of test ids
    test_df = create_df(TEST_IMAGE_PATH)
    X_test = test_df['id'].to_numpy()

    # normalization constants (should match notebook)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # create test dataset
    # Note: dataset.BCSSTestDataset in this repo returns PIL Image or (img, filename)
    test_set = BCSSTestDataset(TEST_IMAGE_PATH, X_test, transform=None)

    # load model
    model = AttentionUNet(img_ch=3, output_ch=3)
    # note: notebook used torch.load('Unet-Resnet.pt') then load_state_dict
    # some checkpoints may be saved with whole model; try both ways
    ckpt_path = 'AttentionUNet_best_miou.pt'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        # prefer state_dict load if checkpoint is a state_dict
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and 'state_dict' in state:
            model.load_state_dict(state['state_dict'])
        elif isinstance(state, dict) and all(k.startswith('module.') or k in model.state_dict() for k in state.keys()):
            model.load_state_dict(state)
        else:
            # fallback: assume checkpoint is a whole model
            model = state
    except Exception:
        # fallback: try to load directly as model
        model = torch.load(ckpt_path, map_location=device)

    model = model.to(device)

    # prediction loop
    data = []
    for i in tqdm(range(len(test_set))):
        img = test_set[i]
        # BCSSTestDataset in notebook returned (img, filename) for test set variant
        # but in our dataset file BCSSTestDataset returns PIL Image; handle both
        if isinstance(img, tuple) or isinstance(img, list):
            img_tensor, filename = img
        else:
            # no filename returned; use index from X_test
            img_tensor = img
            filename = X_test[i]

        pred_mask = predict_image(model, img_tensor, device)

        data.append({'index': filename, 'pred_mask': pred_mask.numpy().tolist()})

    df = pd.DataFrame(data)
    df.to_csv('output.csv', index=False)


if __name__ == '__main__':
    main()
