from torch import nn
from transformers import AutoModel

class ImgEmbedding(nn.Module):
    
    def __init__(self,vision_model_name):
        super(ImgEmbedding, self).__init__()
        self.encoder_img = AutoModel.from_pretrained(vision_model_name)

    def forward(self,pixel_values,pixel_mask=None):
        image_embeds = self.encoder_img(pixel_values).last_hidden_state
        return image_embeds