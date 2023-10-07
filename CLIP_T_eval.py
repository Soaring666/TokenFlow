import clip
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import copy

class CLIPEvaluator(object):
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        self.device = device
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)

        self.clip_preprocess = clip_preprocess
        
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (generator output) to [0, 1].
                                              clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                              clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    @torch.no_grad()
    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def get_text_features(self, text: str, norm: bool = True) -> torch.Tensor:

        tokens = clip.tokenize(text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def img_to_img_similarity(self, src_images, generated_images):
        src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)

        return (src_img_features @ gen_img_features.T).mean()

    def txt_to_img_similarity(self, text, generated_images):
        text_features    = self.get_text_features(text)
        gen_img_features = self.get_image_features(generated_images)

        return (text_features @ gen_img_features.T).mean()


class ImageDirEvaluator(CLIPEvaluator):
    def __init__(self, device, clip_model='ViT-B/32') -> None:
        super().__init__(device, clip_model)

    def evaluate(self, gen_samples, src_images, target_text):

        sim_samples_to_img  = self.img_to_img_similarity(src_images, gen_samples)
        # sim_samples_to_text = self.txt_to_img_similarity(target_text.replace("*", ""), gen_samples)
        sim_samples_to_text = self.txt_to_img_similarity(target_text, gen_samples)

        # print("CLIP-T:", sim_samples_to_text)
        return sim_samples_to_img, sim_samples_to_text

# 将文件夹中的图片制作成一个(n, c, w, h)的tensor
def file_to_tensor(image_folder):
    # 根据文件名进行排序
    file_list = os.listdir(image_folder)
    sorted_file_list = sorted(file_list)
    tensor_list = []

    # 遍历文件夹中的文件
    for filename in sorted_file_list:
        file_path = os.path.join(image_folder, filename)

        # 检查文件是否为图片（可根据需要添加其他图片格式的检查）
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            try:
                image = Image.open(file_path)
                img = np.array(image).astype(np.uint8) 
                img = (img / 127.5 - 1.0).astype(np.float32)
                img = torch.from_numpy(img).permute(2, 0, 1).to(device)
                tensor_list.append(img)
                image.close()
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    file_tensor = torch.stack(tensor_list, axis=0)

    return file_tensor
    
if __name__ == "__main__":
    device = "cuda"
    evaluator = ImageDirEvaluator(device=device)
    prompt = "a wonder woman running"

    # 指定要读取的文件夹路径
    current_folder = os.getcwd()
    src_images = os.path.join(current_folder, "data/woman-running-40")
    gen_images = os.path.join(current_folder, "tokenflow-results_pnp_SD_1.5/woman-running/a wonder woman running, in the space/attn_0.5_f_0.8/batch_size_8/50/img_ode")

    src = file_to_tensor(src_images)        #(8, 3, 804, 804)
    # src = src.float()
    gen = file_to_tensor(gen_images)
    # gen = gen.float()

    sim_img, sim_txt = evaluator.evaluate(gen, src, prompt)
    print("frame_consistency:", sim_img)
    print("CLIP-T:", sim_txt)

