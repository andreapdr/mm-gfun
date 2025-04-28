from tqdm import tqdm

from os import makedirs
import torch

class GFunMM:
    def __init__(
            self,
            text_model,
            vision_model,
            metaclassifier,
            classification_type,
            device="cuda"
            ):
        self.text_model = text_model
        self.vision_model = vision_model
        self.metaclassifier = metaclassifier
        self.classification_type = classification_type

        self.device = device
    
    def first_forward(self, dataloader, cache=False, debug=False):
        print(f"{dataloader.dataset._fingerprint=}")
        vembeds = []
        tembeds = []
        with torch.no_grad():
            for i, (batch, metadata_batch) in enumerate(tqdm(dataloader)):
                if self.device == "cuda":
                    for k, v in batch.items():
                        batch[k] = v.to(self.device)

                label = batch.pop("labels")
        
                is_multimodal = True if "pixel_values" in batch else False
                if is_multimodal:
                    visual_input = {"pixel_values": batch.pop("pixel_values")} 
                    visual_embed = self.vision_model(**visual_input)
                    vembeds.append(visual_embed)
        
                textual_input = batch
                textual_embed = self.text_model(**textual_input)
                tembeds.append(textual_embed)

                if debug:
                    if i >= 5:
                        break
        vembeds = torch.cat(vembeds, dim=0).to("cpu")
        tembeds = torch.cat(tembeds, dim=0).to("cpu")
        
        if cache:
            print("caching computed representations...")
            raise NotImplementedError
        return batch
    
    def second_forward(self, data):
        return data
    
    def _normalize(self, x):
        return x
    
    def _aggregate(self, x):
        return x