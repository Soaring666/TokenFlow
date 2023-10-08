from util import img_embedding
import torch
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

device = "cuda"

pretrained_model_path = "../make-a-protagonist/Make-A-Protagonist/checkpoints/stable-diffusion-2-1-unclip-small" 
img_path = "data/cxk/00000.png"

feature_extractor = CLIPImageProcessor.from_pretrained(pretrained_model_path, subfolder="feature_extractor")
image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_model_path, subfolder="image_encoder")

with torch.no_grad():
    img_emb = img_embedding(img_path, feature_extractor, image_encoder).to(device)
print(img_emb.shape)