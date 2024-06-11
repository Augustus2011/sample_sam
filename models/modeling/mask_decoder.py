import torch
from torch import nn
from torch.nn import functional as F

from typing import List,Tuple,Type,Optional,Type

from .common import LayerNorm2d,Adapter
from .vit import TransformerEncoder
from einops import rearrange

class SmallDecoder(nn.Module):
    def __init__(self,input_chans=256,prompt_embed_dim:int=256,img_size:Optional[Tuple[int,int]]=(256,256),patch_size:int=1,scale:float=256**-0.5,
                 activation:Type[nn.Module]=nn.GELU,depth:int=1,n_cls:int=1,)->None:
        super().__init__()
        self.scale=scale
        self.n_cls=n_cls
        self.img_size=img_size
        self.patch_size=patch_size

        self.cls_emb=nn.Parameter(torch.rand([1,n_cls,prompt_embed_dim]))
        self.dec_proj=nn.Linear(prompt_embed_dim,prompt_embed_dim)

        self.decoder_norm=nn.LayerNorm(prompt_embed_dim)
        self.mask_norm=nn.LayerNorm(n_cls)

        self.proj_patch=nn.Parameter(
            self.scale*torch.randn(prompt_embed_dim,prompt_embed_dim)
        )
        self.proj_classes=nn.Parameter(
            self.scale*torch.randn(prompt_embed_dim,prompt_embed_dim)
        )
        self.blocks=TransformerEncoder(depth=depth)
        self.upsampling=nn.Sequential(
            nn.ConvTranspose2d(prompt_embed_dim,prompt_embed_dim,kernel_size=2,stride=2),
            LayerNorm2d(prompt_embed_dim),
            activation(),
            nn.ConvTranspose2d(prompt_embed_dim,prompt_embed_dim,kernel_size=2,stride=2),
            activation(),
        )

    def forward(self,image_embedding:torch.Tensor)->torch.Tensor:
        b,c,h,w=image_embedding.shape
        image_embedding=image_embedding.flatten(2).permute(0,2,1)
        H,W=self.img_size
        GS=H//self.patch_size
        x=self.dec_proj(image_embedding)
        
        cls_emb=self.cls_emb.expand(x.size(0),-1,-1)
        x=torch.cat((x,cls_emb),1)
        out=(self.blocks(x))
        x=self.decoder_norm(out)

        patches,cls_seg_feat=x[:,:-self.n_cls],x[:,-self.n_cls:]
        patches=patches.transpose(1,2).view(b,c,h,w)
        patches=self.upsampling(patches)
        
        patches=patches@self.proj_patch
        cls_seg_feat=cls_seg_feat@self.proj_classes

        patches=patches/patches.norm(dim=-1,keepdim=True)
        cls_seg_feat=cls_seg_feat/cls_seg_feat.norm(dim=-1,keepdim=True)
        masks=patches@cls_seg_feat.transpose(1,2)
        masks=rearrange(masks,"b (h w) n -> b n h w",h=int(GS))
        out=masks
        return out
    
class MaskDecoder(nn.Module):
    def __init__(self, *,transformer_dim:int,transformer:nn.Module,num_multimask_outputs:int=3,activation:Type[nn.Module]=nn.GELU,
                 iou_head_depth:int=3,iou_head_hidden_dim:int=256) -> None:
        super().__init__()
        
        self.transformer_dim=transformer_dim
        self.transformer=transformer
        self.num_multimask_outputs=num_multimask_outputs
        self.iou_token=nn.Embedding(1,transformer_dim)
        self.num_mask_tokens=num_multimask_outputs+1
        self.mask_tokens=nn.Embedding(self.num_mask_tokens,transformer_dim)
        
        self.output_upscaling=nn.Sequential(
            nn.ConvTranspose2d(transformer_dim,transformer_dim//4,kernel_size=2,stride=2),
            LayerNorm2d(transformer_dim//4),
            activation(),
            nn.ConvTranspose2d(transformer_dim//4,transformer_dim//8,kernel_size=2,stride=2),
            activation(),
        )
        
        self.output_hypernetworks_mlps=nn.ModuleList(
            [
                MLP(transformer_dim,transformer_dim,transformer_dim//8,3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head=MLP(
            transformer_dim,iou_head_hidden_dim,self.num_mask_tokens,iou_head_depth
        )

    def forward(self,image_embeddings:torch.Tensor,image_pe:torch.Tensor,
                sparse_prompt_embeddings:torch.Tensor,dense_prompt_embeddings:torch.Tensor,
                multimask_output:bool)->Tuple[torch.Tensor,torch.Tensor]:
        masks,iou_pred=self.predict_masks(image_embeddings=image_embeddings,image_pe=image_pe,
                                          sparse_prompt_embeddings=sparse_prompt_embeddings,
                                          dense_prompt_embeddings=dense_prompt_embeddings,
                                          )
        if multimask_output:
            mask_slice=slice(1,None)
        else:
            mask_slice=slice(0,1)
        masks=masks[:,mask_slice,:,:]
        iou_pred=iou_pred[:,mask_slice]

        return masks,iou_pred


    def predict_masks(self,image_embeddings:torch.Tensor,image_pe:torch.Tensor,
                      sparse_prompt_embeddings:torch.Tensor,dense_prompt_embeddings:torch.Tensor,)->Tuple[torch.Tensor,torch.Tensor]:
        output_tokens=torch.cat([self.iou_token.weight,self.mask_tokens],dim=0)
        output_tokens=output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0),-1,-1)
        tokens=torch.cat((output_tokens,sparse_prompt_embeddings),dim=1)
        
        if image_embeddings.shape[0]!=tokens.shape[0]:
            src=torch.repeat_interleave(image_embeddings,tokens.shape[0],dim=0)
        else:
            src=image_embeddings
        src=src+dense_prompt_embeddings
        pos_src=torch.repeat_interleave(image_pe,tokens.shape[0],dim=0)
        b,c,h,w=src.shape

        hs,src=self.transformer(src,pos_src,tokens)
        iou_token_out=hs[:,0,:]
        mask_tokens_out=hs[:,1:(1+self.num_mask_tokens),:]
        src=src.transpose(1,2).view(b,c,h,w)
        upscaled_embedding=self.output_upscaling(src)
        hyper_in_list:List[torch.Tensor]=[]
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:,i,:]))
        hyper_in=torch.stack(hyper_in_list,dim=1)
        b,c,h,w=upscaled_embedding.shape
        masks=(hyper_in@upscaled_embedding.view(b,c,h*w)).view(b,-1,h,w)
        iou_pred=self.iou_prediction_head(iou_token_out)

        return masks,iou_pred


class MLP(nn.Module):
    def __init__(self,input_dim:int,hidden_dim:int,output_dim:int,num_layers:int,sigmoid_output:bool=False)->None:
        super().__init__()
        self.num_layers=num_layers
        h=[hidden_dim]*(num_layers-1)
        self.layers=nn.ModuleList(
            nn.Linear(n,k) for n,k in zip([input_dim]+h,h+[output_dim])
        )
        self.sigmoid_output=sigmoid_output
    def forward(self,x:torch.Tensor)->torch.Tensor:
        for i,layer in enumerate(self.layers):
            x=F.relu(layer(x)) if i <self.num_layers-1 else layer(x)
        if self.sigmoid_output:
            x=F.sigmoid(x)
        return x

