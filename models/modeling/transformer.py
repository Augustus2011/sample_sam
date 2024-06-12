

import torch
from torch import Tensor,nn

import math
from typing import Tuple,Type,Optional

from common import MLPBlock,Adapter


class TwoWayTransformer(nn.Module):
    def __init__(self,args,depth:int,embedding_dim:int,num_heads:int,mlp_dim:int,
                 activation:Type[nn.Module]=nn.ReLU,attention_downsample_rate:int=2)->None:
        super().__init__()
        self.args=args
        self.depth=depth
        self.embedding_dim=embedding_dim
        self.num_heads=num_heads
        self.mlp_dim=mlp_dim
        self.layers=nn.ModuleList()
        
        for i in range(depth):
            if i<args.decoder_adapt_depth:
                if_adapter=args.if_mask_decoder_adapter
            else:
                if_adapter=False
            self.layers.append(
                TwoWayAttentionBlock(args=self.args,embedding_dim=embedding_dim,
                                     num_heads=num_heads,mlp_dim=mlp_dim,
                                     activation=activation,attention_downsample_rate=attention_downsample_rate,
                                     if_adapter=if_adapter,skip_first_layer_pe=(i==0),
                                     )
            )
        self.final_attn_token_to_image=Attention(embedding_dim=embedding_dim,num_heads=num_heads,
                                                 downsample_rate=attention_downsample_rate)
        self.norm_final_attn=nn.LayerNorm(embedding_dim)
    
    def forward(self,image_embedding:Tensor,image_pe:Tensor,point_embedding:Tensor)->Tuple[Tensor,Tensor]:
        bs,c,h,w=image_embedding.shape
        image_embedding=image_embedding.flatten(2).permute(0,2,1)
        image_pe=image_pe.flatten(2).permute(0,2,1)
        
        queries=point_embedding
        keys=image_embedding
        
        for layer in self.layers:
            queries,keys =layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )
        
        q=queries+point_embedding
        k=keys+image_pe
        attn_out=self.final_attn_token_to_image(q=q,k=k,v=keys)
        queries=queries+attn_out
        queries=self.norm_final_attn(queries)

        return queries,keys

class TwoWayAttentionBlock(nn.Module):
    def __init__(self,args,embedding_dim:int,num_heads:int,mlp_dim:int=2048,activation:Type[nn.Module]=nn.ReLU,
                 attention_downsample_rate:int=2,if_adapter:bool=False,skip_first_layer_pe:bool=False)->None:
        super().__init__()
        self.args=args
        self.if_adapter=if_adapter
        self.self_attn=Attention(embedding_dim,num_heads)
        self.norm1=nn.LayerNorm(embedding_dim)
        
        self.cross_attn_token_to_image=Attention(embedding_dim,num_heads,attention_downsample_rate)
        self.norm2=nn.LayerNorm(embedding_dim)

        self.mlp=MLPBlock(embedding_dim,mlp_dim,activation)
        self.norm3=nn.LayerNorm(embedding_dim)
        self.norm4=nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image=Attention(embedding_dim,num_heads,attention_downsample_rate)

        if self.if_adapter:
            self.MLP_Adapter=Adapter(embedding_dim,skip_connect=False)
            self.Adapter=Adapter(embedding_dim)
            self.scale=0.5
        self.skip_first_layer_pe=skip_first_layer_pe

    def forward(self,queries:Tensor,keys:Tensor,query_pe:Tensor,key_pe:Tensor)->Tuple[Tensor,Tensor]:
        if self.skip_first_layer_pe:
            queries=self.self_attn(q=queries,k=queries,v=queries)
        else:
            q=queries+query_pe
            attn_out=self.self_attn(q=q,k=q,v=queries)
            queries=queries+attn_out
        queries=self.norm1(queries)
        #cross attenion block
        q=queries+query_pe
        k=keys+key_pe
        attn_out=self.cross_attn_token_to_image(q=q,k=k,v=keys)
        queries=queries+attn_out
        
        if self.if_adapter:
            queries=self.norm2(queries)

        mlp_out=self.mlp(queries)
        if self.if_adapter:
            queries=queries+mlp_out+self.scale*self.MLP_Adapter(queries)
        else:
            queries=queries+mlp_out
        queries=self.norm3(queries)

        q=queries+query_pe
        k=keys+key_pe
        attn_out=self.cross_attn_token_to_image(q=k,k=q,v=queries)
        keys=keys+attn_out

        if self.if_adapter:
            keys=self.Adapter(keys)
        keys=self.norm4(keys)
        return queries,keys

class Attention(nn.Module):
    def __init__(self,embedding_dim:int,num_heads:int,downsample_rate:int=1,)->None:
        super().__init__()
        self.embedding_dim=embedding_dim
        self.internal_dim=embedding_dim//downsample_rate
        self.num_heads=num_heads
        assert self.internal_dim % num_heads ==0, "num_heads must divisible by num_heads"
        
        self.q_proj=nn.Linear(embedding_dim,self.internal_dim)
        self.k_proj=nn.Linear(embedding_dim,self.internal_dim)
        self.v_proj=nn.Linear(embedding_dim,self.internal_dim)
        
        self.out_proj=nn.Linear(self.internal_dim,embedding_dim)
        
    def _separete_heads(self,x:Tensor,num_heads:int)->Tensor:
        b,n,c=x.shape
        x=x.reshape(b,n,num_heads,c//num_heads)
        return x.transpose(1,2)
    
    def _recombine_heads(self,x:Tensor)->Tensor:
        b,n_heads,n_tokens,c_per_head=x.shape
        x=x.transpose(1,2)
        return x.reshape(b,n_tokens,n_heads*c_per_head)
    
    def forward(self,q:Tensor,k:Tensor,v:Tensor)->Tensor:
        q=self.q_proj(q)
        k=self.k_proj(k)
        v=self.v_proj(v)
        
        q=self._separete_heads(q,self.num_heads)
        k=self._separete_heads(k,self.num_heads)
        v=self._separete_heads(v,self.num_heads)
        
        _,_,_,c_per_head=q.shape
        attn=q@k.permute(0,1,3,2)
        attn=attn/math.sqrt(c_per_head)
        attn=torch.softmax(attn,dim=-1)
        
        out=attn@v
        out=self._recombine_heads(out)
        out=self.out_proj(out)
        
        return out