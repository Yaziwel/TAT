import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange
import math
def default_conv(in_channels, out_channels, kernel_size, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.GELU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        # self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x

        return res

class TREN(nn.Module):
    def __init__(self, n_feats=64, n_encoder_res=2, dim=48):
        super(TREN, self).__init__()
        E1 = [nn.Conv2d(24, n_feats, kernel_size=3, padding=1),
              nn.GELU()]
        E2 = [
            ResBlock(
                default_conv, n_feats, kernel_size=3
            ) for _ in range(n_encoder_res)
        ]
        E3 = [
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(n_feats*2, n_feats * 4, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        ]
        E = E1 + E2 + E3
        self.E = nn.Sequential(
            *E
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_feats * 4, n_feats * 4, bias=False),
            nn.GELU(),
            nn.Linear(n_feats * 4, n_feats * 4, bias=False),
            nn.GELU()
        )
        self.pixel_unshuffle = nn.PixelShuffle(4)

        self.kernel_latent_ffn =  nn.Sequential(
            nn.Linear(256, int(dim*(2**3)*2.66)*2*9, bias=False),
            nn.GELU()
        )

        self.kernel_latent_attn =  nn.Sequential(
            nn.Linear(256, dim*(2**3)*3*9, bias=False),
            nn.GELU()
        )
        self.kernel_level3_ffn =  nn.Sequential(
            nn.Linear(256, int(dim*(2**2)*2.66)*2*9, bias=False),
            nn.GELU()
        )
        self.kernel_level3_attn =  nn.Sequential(
            nn.Linear(256, dim*(2**2)*3*9, bias=False),
            nn.GELU()
        )
        self.kernel_level2_ffn =  nn.Sequential(
            nn.Linear(256, int(dim*(2**1)*2.66)*2*9, bias=False),
            nn.GELU()
        )
        self.kernel_level2_attn =  nn.Sequential(
            nn.Linear(256, dim*(2**1)*3*9, bias=False),
            nn.GELU()
        )
        self.kernel_level1_ffn =  nn.Sequential(
            nn.Linear(256, int(dim*(2**1)*2.66)*2*9, bias=False),
            nn.GELU()
        )
        self.kernel_level1_attn =  nn.Sequential(
            nn.Linear(256, dim*(2**1)*3*9, bias=False),
            nn.GELU()
        )
    

    def forward(self, x): 
        # import pdb 
        # pdb.set_trace() 
        x = x.clone().detach()
        x = self.pixel_unshuffle(x)
        fea = self.E(x).squeeze(-1).squeeze(-1)
        fea1 = self.mlp(fea)
        return self.kernel_level1_ffn(fea1).view(x.shape[0],-1,1,3,3),  self.kernel_level1_attn(fea1).view(x.shape[0],-1,1,3,3), self.kernel_level2_ffn(fea1).view(x.shape[0],-1,1,3,3), self.kernel_level2_attn(fea1).view(x.shape[0],-1,1,3,3), self.kernel_level3_ffn(fea1).view(x.shape[0],-1,1,3,3),  self.kernel_level3_attn(fea1).view(x.shape[0],-1,1,3,3), self.kernel_latent_ffn(fea1).view(x.shape[0],-1,1,3,3), self.kernel_latent_attn(fea1).view(x.shape[0],-1,1,3,3), 

def to(x):
    return {'device': x.device, 'dtype': x.dtype}

def pair(x):
    return (x, x) if not isinstance(x, tuple) else x


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward_encoder(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_encoder, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class FeedForward_decoder(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward_decoder, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        
        
        self.dw_kernel = nn.Parameter(torch.empty(1, hidden_features * 2, 1, 3, 3)) 
        nn.init.kaiming_normal_(self.dw_kernel, mode='fan_out', nonlinearity='relu')
        self.lambda_dw = nn.Parameter(torch.zeros(1, hidden_features*2, 1, 1, 1)) 

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.dim = dim

        self.hidden_features = hidden_features

    def forward(self, x, gen_kernel):
        
        x = self.project_in(x) 
        B, C, H, W = x.shape

        x = x.view(1, B * C, H, W) 
        kernel = self.dw_kernel + self.lambda_dw*gen_kernel 
        kernel = kernel.view(B * C, 1, 3, 3) 
        x = F.conv2d(x, kernel, stride=1, padding=1, groups=B*C)
        x = x.view(B, C, H, W) 
        
        
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention_encoder(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_encoder, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        self.train_size = (128//num_heads, 128//num_heads) 
        self.kernel_size = (int(1.5*128//num_heads), int(1.5*128//num_heads)) 

    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c//3, h, w)
        assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt

    def forward(self, x):

        qkv = self.qkv_dwconv(self.qkv(x)) 
        # import pdb 
        # pdb.set_trace()
        
        use_TLC = (x.shape[2], x.shape[3])!=self.train_size
        if use_TLC:
            qkv = self.grids(qkv) # convert to local windows 
        q,k,v = qkv.chunk(3, dim=1) 

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=qkv.shape[-2], w=qkv.shape[-1])
        if use_TLC:
            out = self.grids_inverse(out) # reverse 

        out = self.project_out(out)
        return out 


class Attention_decoder(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention_decoder, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        # self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias) 

        self.dw_kernel = nn.Parameter(torch.randn(1, dim*3, 1, 3, 3)) 
        # nn.init.kaiming_normal_(self.dw_kernel, mode='fan_out', nonlinearity='relu')
        self.lambda_dw = nn.Parameter(torch.zeros(1, dim*3, 1, 1, 1)) 
        self.dim=dim
        
        self.train_size = (128//num_heads, 128//num_heads) 
        self.kernel_size = (int(1.5*128//num_heads), int(1.5*128//num_heads)) 

    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c//3, h, w)
        # assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt

    def forward(self, x, gen_kernel):

        x = self.qkv(x) 
        
        B, C, H, W = x.shape
        x = x.view(1, B * C, H, W) 
        # import pdb 
        # pdb.set_trace()
        kernel = self.dw_kernel + self.lambda_dw*gen_kernel 
        kernel = kernel.view(B * C, 1, 3, 3) 
        x = F.conv2d(x, kernel, stride=1, padding=1, groups=B*C)
        qkv = x.view(B, C, H, W)
        
        use_TLC = (qkv.shape[2], qkv.shape[3])!=self.train_size 
        # import pdb 
        # pdb.set_trace()
        if use_TLC:
            qkv = self.grids(qkv) # convert to local windows 
        q,k,v = qkv.chunk(3, dim=1) 

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v) 
        # import pdb 
        # pdb.set_trace()
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', h=qkv.shape[-2], w=qkv.shape[-1])
        if use_TLC:
            out = self.grids_inverse(out) # reverse 

        out = self.project_out(out)
        return out

##########################################################################
##########################################################################
class TransformerBlock_encoder(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_encoder, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention_encoder(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward_encoder(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

##########################################################################
class TransformerBlock_decoder(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock_decoder, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention_decoder(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward_decoder(dim, ffn_expansion_factor, bias)

    def forward(self, y):
        x = y[0]      
        x = x + self.attn(self.norm1(x), y[2])
        x = x + self.ffn(self.norm2(x), y[1])
        return [x, y[1], y[2]]


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=1, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- Restormer -----------------------
class TAT(nn.Module):
    def __init__(self, 
        inp_channels=1, 
        out_channels=1, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
    ):

        super(TAT, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock_encoder(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock_encoder(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock_encoder(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock_decoder(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        
        self.up4_3 = Upsample(int(dim*2**3)) ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock_decoder(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock_decoder(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock_decoder(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        
        self.refinement = nn.Sequential(*[TransformerBlock_decoder(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])
        
        self.reduce_final = nn.Sequential( 
            nn.Conv2d(dim*5, dim, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.GELU(), 
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias), 
            )
        

            
        self.output = nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.tren = TREN()

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1= self.encoder_level1(inp_enc_level1)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)

        kernel = self.tren(inp_enc_level4)

        latent, _, _ = self.latent([inp_enc_level4, kernel[6], kernel[7]])

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3, _, _ = self.decoder_level3([inp_dec_level3, kernel[4], kernel[5]])

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2, _, _ = self.decoder_level2([inp_dec_level2, kernel[2], kernel[3]])

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1, _, _ = self.decoder_level1([inp_dec_level1, kernel[0], kernel[1]])

        out_dec_level1_refine, _, _ = self.refinement([out_dec_level1, kernel[0], kernel[1]]) 

        out = torch.cat([out_enc_level1, out_dec_level1, out_dec_level1_refine], dim=1) 
        out = self.reduce_final(out) 

        out = self.output(out) + inp_img

        return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 

# model = MedRWKV()
# num = count_parameters(model) 
# print(num/1e6)
if __name__ == "__main__":
    import os 
    os.environ['CUDA_VISIBLE_DEVICES']='1' 
    # x=torch.zeros((1,3,513,513)).type(torch.FloatTensor).cuda() 
    
    import time 
    
    # # y = mapping(x)
    # G=IPT() 
    # G.cuda()
    # with torch.no_grad():
    #     y=G(x) 
    # # print(time.time()-since) 
    from thop import profile, clever_format
    
    x=torch.zeros((1,1,128, 128)).type(torch.FloatTensor).cuda() 
    model = TAT() 
    # print(model)
    model.cuda() 
    
    since = time.time()
    y=model(x)
    print("time", time.time()-since) 
    
    flops, params = profile(model, inputs=(x, ))  
    flops, params = clever_format([flops, params], '%.3f') 
    print('flops',flops)
    print('params', params) 
    print(count_parameters(model)/1e6)
    # print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
    # print("Params=", str(params/1e6)+'{}'.format("M"))