import numpy as np

import torch
from .modeling import Sam
from typing import Optional,Tuple

from .modeling.utils.transforms import ResizeLongestSide

class SamPredictor:
    def __init__(self,sam_model:Sam,)->None:
        super().__init__()
        self.model=sam_model
        self.transform=ResizeLongestSide(sam_model.image_encoder.img_size)
        self.reset_image()

    def set_image(self,image:np.ndarray,image_format:str="RGB",)->None:
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]
        
        input_image=self.transform.apply_image(image)
        input_image_torch=torch.as_tensor(input_image,device=self.device)
        input_image_torch=input_image_torch.permute(2,0,1).contiguous()[None,:,:,:]

        self.set_torch_image(input_image_torch,image.shape[:2])

    @torch.no_grad()
    def set_torch_image(self,transformed_image:torch.Tensor,original_image_size:Tuple[int,...],)->None:
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be BCHW with long side {self.model.image_encoder.img_size}."
        
        self.reset_image()
        self.original_size=original_image_size
        self.input_size=tuple(transformed_image.shape[-2:])
        input_image=torch.tensor(transformed_image*1.0/255)
        self.features=self.model.image_encoder(input_image)
        self.is_image_set=True

    def predict(self,point_coords:Optional[np.ndarray]=None,point_labels:Optional[np.ndarray]=None,
                box:Optional[np.ndarray]=None,mask_input:Optional[np.ndarray]=None,multimask_output:bool=True,
                return_logits:bool=False)->Tuple[np.ndarray,np.ndarray,np.ndarray]:
        
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")
        
        coords_torch,labels_torch,box_torch,mask_input_torch=None,None,None,None

        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."

            point_coords=self.transform.apply_coords(point_coords,self.original_size)
            coords_torch=torch.as_tensor(point_coords,dtype=torch.float,device=self.device)
            labels_torch=torch.as_tensor(point_labels,dtype=torch.float,device=self.device)
            coords_torch,labels_torch=coords_torch[None,:,:],labels_torch[None,:]

        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        masks,iou_predictions,low_res_masks=self.predict_torch(coords_torch,labels_torch,box_torch,mask_input_torch,multimask_output,return_logits,)

        mask_np=masks[0].detach().cpu().numpy()
        iou_predictions=iou_predictions[0].detach().cpu().numpy()
        low_res_masks=low_res_masks[0].detach().cpu().numpy()
        return mask_np,iou_predictions,low_res_masks




    @torch.no_grad()
    def predict_torch(self,point_coords:Optional[torch.Tensor],point_labels:Optional[torch.Tensor],
                      boxes:Optional[torch.Tensor]=None,mask_input:Optional[torch.Tensor]=None,
                      multimask_output:bool=True,return_logits:bool=False)->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        if not self.is_image_set:
            raise RuntimeError("an image must set witn .set_image")
        if point_coords is not None:
            points=(point_coords,point_labels)
        else:
            points=None
        
        sparse_embeddings,dense_embeddings=self.model.prompt_encoder(
            points=points,boxes=boxes,masks=mask_input,
        )

        low_res_masks,iou_predictions=self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        masks=self.model.postprocess_mask(low_res_masks,self.input_size,self.original_size)

        if not return_logits:
            masks=masks>self.model.mask_threshold
        
        return masks,iou_predictions,low_res_masks
    
    def get_image_embedding(self)->torch.Tensor:
        if not self.is_image_set:
            raise RuntimeError("an image must be set with .set_image")
        
        assert self.features is not None, "Features must exist if an image has been set."
        return self.features
    
    @property
    def device(self)->torch.device:
        return self.model.device
    
    def reset_image(self)->None:
        self.is_image_set=False
        self.features=None
        self.orig_h=None
        self.orig_w=None
        self.input_h=None
        self.input_w=None