import argparse, os
from tqdm import trange
import torch
from einops import rearrange
from omegaconf import OmegaConf
import json
from run_features_extraction import load_model_from_config
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from PIL import Image
from math import sqrt
from torchvision import transforms as T

from ldm.models.diffusion.ddim import DDIMSampler


def load_experiments_features(feature_maps_paths, t):
    feature_maps = []
    for i, feature_path in enumerate(feature_maps_paths):
        feature_map = torch.load(os.path.join(feature_path, f"down_blocks2_resnets1_timestep_{t}.pt"))[0]
        feature_map = feature_map.reshape(feature_map.shape[0], -1).t()  # N X C
        feature_maps.append(feature_map)

    return feature_maps

def visualize_and_save_features_pca(feature_maps_fit_data,feature_maps_transform_data, t, save_dir):
    feature_maps_fit_data = feature_maps_fit_data.cpu().numpy()
    pca = PCA(n_components=3)
    pca.fit(feature_maps_fit_data)
    feature_maps_pca = pca.transform(feature_maps_transform_data.cpu().numpy())  # N X 3
    feature_maps_pca = feature_maps_pca.reshape(1, -1, 3)  # B x (H * W) x 3

    pca_img = feature_maps_pca[0]  # (H * W) x 3
    h = w = int(sqrt(pca_img.shape[0]))
    pca_img = pca_img.reshape(h, w, 3)
    pca_img_min = pca_img.min(axis=(0, 1))
    pca_img_max = pca_img.max(axis=(0, 1))
    pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
    pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
    pca_img = T.Resize(512, interpolation=T.InterpolationMode.NEAREST)(pca_img)
    pca_img.save(os.path.join(save_dir, f"time_{t}.png"))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_config",
        type=str,
        default="/home/flyvideo/PCH/diffusion/plug-and-play/configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/home/flyvideo/PCH/diffusion/plug-and-play/sd_1.4/sd-v1-4.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--feature_path",
        type=str,
        default="pnp_cond_image_feature_extract/wt",
        help="path of extracted feature",
    )

    opt = parser.parse_args()

    ddim_steps = 50
    model_config = OmegaConf.load(f"{opt.model_config}")
    model = load_model_from_config(model_config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=0, verbose=False)
    time_range = np.flip(sampler.ddim_timesteps)
    total_steps = sampler.ddim_timesteps.shape[0]
    iterator = tqdm(time_range, desc="visualizing features", total=total_steps)

    print("visualizing features PCA experiments")

    transform_feature_maps_paths = []
    transform_feature_maps_paths.append(opt.feature_path)
    fit_feature_maps_paths = []
    fit_feature_maps_paths.append(opt.feature_path)

    pca_folder_path = os.path.join("pnp_cond_image_feature_PCA_vis", os.path.basename(opt.feature_path))
    os.makedirs(pca_folder_path, exist_ok=True)

    for t in iterator:
        fit_features = load_experiments_features(fit_feature_maps_paths, t)  # N X C
        transform_features = load_experiments_features(transform_feature_maps_paths, t)
        visualize_and_save_features_pca(torch.cat(fit_features, dim=0),
                                        torch.cat(transform_features, dim=0),
                                        t,
                                        pca_folder_path)

    print("Done!")

if __name__ == "__main__":
    main()
