import torch
import torch.nn as nn
import torch.nn.functional as F
from util import sample_and_group 
from einops.layers.torch import Rearrange
from einops import reduce, rearrange, repeat
import numpy as np

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Local_op, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 6]) 
        x = x.permute(0, 1, 3, 2)   
        x = x.reshape(-1, d, s) 
        batch_size, _, N = x.size()
        x = F.relu(self.bn1(self.conv1(x))) # B, D, N
        x = F.relu(self.bn2(self.conv2(x))) # B, D, N
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x

class DropPredictor(nn.Module):
    """ Computes the log-probabilities of dropping a token, adapted from PredictorLG here:
    https://github.com/raoyongming/DynamicViT/blob/48ac52643a637ed5a4cf7c7d429dcf17243794cd/models/dyvit.py#L287 """
    def __init__(self, embed_dim):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, policy):
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:,:, :C//2]
        global_x = (x[:,:, C//2:] * policy).sum(dim=1, keepdim=True) / (torch.sum(policy, dim=1, keepdim=True)+0.000001)
        x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=-1)
        return self.out_conv(x)

class Pct(nn.Module):
    def __init__(self, args, output_channels=40, adaptive = False, layers_to_drop = []):
        super(Pct, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        if adaptive:   
            self.pt_last = Point_Transformer_Adaptive(args, channels=256, d_model=512, d_k=32, d_v=64, n_heads=8, n_blocks=4, layers_to_drop=layers_to_drop)
        else:
            self.pt_last = Point_Transformer_Last(args, channels=256, d_model=512, d_k=32, d_v=64, n_heads=8, n_blocks=4)
        



        self.linear1 = nn.Linear(512, 256, bias=False)
        self.bn6 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(256, 128)
        self.bn7 = nn.BatchNorm1d(128)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(128, output_channels)

    def forward(self, x, drop_temp=1):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))
        
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)
        
        #print("feature0:", feature_0.shape)
        
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)

        #print("feature1:", feature_1.shape)

        x, masks,distr = self.pt_last(feature_1, drop_temp=drop_temp)
        #x, masks = self.pt_last(x)

        #print("output SA:", x.shape)

        #POOLING INSTEAD OF CLASS TOKEN
        x = F.adaptive_max_pool1d(x[:,:-1,:], 2).view(batch_size, -1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)

        #x = F.leaky_relu(self.bn6(self.linear1(x[:,-1,:])), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)

        return x, masks, distr

class Point_Transformer_Last(nn.Module):
    def __init__(self, args, channels=256, d_model = 256, d_k=16, d_v=32, n_heads=8, n_blocks=4):
        super(Point_Transformer_Last, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(channels, d_model, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(d_model)
        self.bn2 = nn.BatchNorm1d(d_model)

        self.cls_token = nn.Parameter(torch.randn((d_model,), requires_grad=True)) # Class token
        self.blocks = nn.ModuleList([TransformerBlock(d_model, d_k, d_v, n_heads) for _ in range(n_blocks)]) # Transformer blocks


    def forward(self, x):
        # 
        # b, 3, npoint, nsample  
        # conv2d 3 -> 128 channels 1, 1
        # b * npoint, c, nsample 
        # permute reshape
        batch_size, _, N = x.size()
        
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = x.permute(0,2,1)
        x = torch.cat([x, repeat(self.cls_token, 'v -> b 1 v', b=x.shape[0])], dim=1)
        
        #print("prima dei blocchi:", x.shape)

        for i, l in enumerate(self.blocks):
          x = l(x)
          #print("dopo_mha:", x.shape)

        return x

class MultiHeadAttentionNew(nn.Module):
    """ Multihead attention from here: https://einops.rocks/pytorch-examples.html 
    Useful if we want to further modify the model """
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head

        self.w_qs = nn.Conv1d(d_model, n_head * d_k, kernel_size=1, bias=False)
        self.w_ks = nn.Conv1d(d_model, n_head * d_k, kernel_size=1, bias=False)
        self.w_vs = nn.Conv1d(d_model, n_head * d_v, kernel_size=1, bias=False)
        
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        
        self.fc = nn.Conv1d(n_head * d_v, d_model, kernel_size=1, bias=False)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(p=dropout)

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        # This is the masked softmax (Eq. (11) in the paper), 
        # taken from here: https://github.com/raoyongming/DynamicViT/blob/master/models/dyvit.py
        B, N, _ = policy.size()
        B, H, N, N = attn.size()

        attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
        eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)
        attn_policy = attn_policy + (1.0 - attn_policy) * eye
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att
        # attn = attn.exp_() * attn_policy
        # return attn / attn.sum(dim=-1, keepdim=True)

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward(self, x, policy=None):

        x = x.permute(0,2,1)

        # x (batch, tokens, features) are the tokens.
        # policy (batch, tokens, 1) is a boolean mask denoting which tokens we should remove from the computation.
        q = rearrange(self.w_qs(x), 'b (head k) t -> b head t k', head=self.n_head)
        k = rearrange(self.w_ks(x), 'b (head k) t -> b head t k', head=self.n_head)
        v = rearrange(self.w_vs(x), 'b (head v) t -> b head t v', head=self.n_head)
        attn = torch.einsum('bhlk,bhtk->bhlt', [q, k]) / np.sqrt(q.shape[-1])
        if policy is None:
            attn = attn.softmax(dim=-1)
        else:
            attn = self.softmax_with_policy(attn, policy)
        output = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        output = rearrange(output, 'b head l v -> b (head v) l')
        output = self.dropout(self.fc(output))
        return output, attn

class TransformerBlock(nn.Module):
  """ A more-or-less standard transformer block. """
  def __init__(self, d_model, d_k, d_v, n_heads, dropout=0.1):
    super().__init__()
    self.sa = MultiHeadAttentionNew(n_heads, d_model, d_k, d_v, dropout=dropout)
    self.ln1 = nn.LayerNorm(d_model)
    self.ln2 = nn.LayerNorm(d_model)
    self.ff = nn.Sequential(
        nn.Linear(d_model, d_model*2),
        nn.GELU(),
        nn.Linear(d_model*2, d_model)
    )

  def forward(self, x, policy=None):
    x = self.sa(self.ln1(x), policy=policy)[0].permute(0,2,1) + x
    x = self.ff(self.ln2(x)) + x
    return x
  


class Point_Transformer_Adaptive(nn.Module):
    def __init__(self, args, channels=256, d_model = 256, d_k=16, d_v=32, n_heads=8, n_blocks=4, layers_to_drop=[]):
        super(Point_Transformer_Adaptive, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(channels, d_model, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(d_model)
        self.bn2 = nn.BatchNorm1d(d_model)

        self.cls_token = nn.Parameter(torch.randn((d_model,), requires_grad=True)) # Class token
        self.blocks = nn.ModuleList([TransformerBlock(d_model, d_k, d_v, n_heads) for _ in range(n_blocks)]) # Transformer blocks
        self.layers_to_drop = layers_to_drop
        self.score_predictor = nn.ModuleList([DropPredictor(d_model) for _ in range(n_blocks)])
        
    def forward(self, x, drop_temp=1):
        
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = x.permute(0,2,1)
        x = torch.cat([x, repeat(self.cls_token, 'v -> b 1 v', b=x.shape[0])], dim=1)
    
        # Initialize drop decisions
        B, P, _ = x.shape
        prev_decision = torch.ones(B, P-1, 1, dtype=x.dtype, device=x.device)
        policy = torch.ones(B, P, 1, dtype=x.dtype, device=x.device)
        #policy = torch.zeros(B, P, 1, dtype=x.dtype, device=x.device)
        out_pred_prob = []
        pred_distr = [[],[],[],[]]

        for i, l in enumerate(self.blocks):
            if i in self.layers_to_drop:
                # Ignore the class token
                points_x = x[:, :-1]
                # Current drop score
                pred_score = self.score_predictor[i](points_x, prev_decision)#.reshape(B, -1, 2)
                #for visualization purposes
                pred_distr[i].append(pred_score.reshape(-1,2))
                # Slow warmup
                keepall = torch.cat((torch.zeros_like(pred_score[:,:,0:1]), torch.ones_like(pred_score[:,:,1:2])),2) 
                pred_score = pred_score*drop_temp + keepall*(1-drop_temp)
                
                if True:#self.training:
                    # Convert to log-prob
                    pred_score = torch.log(pred_score + 1e-8)
                    # Sample mask and update previous one
                    #hard_keep_decision = F.gumbel_softmax(pred_score, hard=True)[:, :, 1:2] * prev_decision
                    # OR Deterministic mask and update previous one
                    soft = F.softmax(pred_score, dim=-1)[:, :, 1:2]
                    hard = torch.argmax(pred_score, dim=-1).float().unsqueeze(-1)
                    decision = soft + (hard - soft).detach()
                    hard_keep_decision = decision * prev_decision
                else:
                    # Treshold mask and update previous one
                    hard_keep_decision = (pred_score[:, :, 1:2] > 0.9).float() * prev_decision
                
                #"Dropout", added for debugging purposes"
                #stuff = (1-self.args.drop_ratio[i])*drop_temp + (1-drop_temp)
                #hard_keep_decision = torch.bernoulli(torch.ones_like(hard_keep_decision)*stuff).float().to(hard_keep_decision.device)
                
                #for visualization purposes
                out_pred_prob.append(hard_keep_decision.reshape(B, P-1))
                # Build the full policy (always keep the class token)
                cls_policy = torch.ones(B, 1, 1, dtype=hard_keep_decision.dtype, device=hard_keep_decision.device)
                policy = torch.cat([cls_policy, hard_keep_decision], dim=1)
                prev_decision = hard_keep_decision
            x = l(x, policy=policy)

        return x, out_pred_prob, pred_distr



class CrossAttention(nn.Module):
    
    def __init__(self, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.w_qs = nn.Conv1d(d_model, d_k, kernel_size=1, bias=False)
        self.w_ks = nn.Conv1d(d_model, d_k, kernel_size=1, bias=False)
        self.w_vs = nn.Conv1d(d_model, d_v, kernel_size=1, bias=False)
        
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))
        
        self.fc = nn.Conv1d(d_v, d_model, kernel_size=1, bias=False)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x, ):

        x = x.permute(0,2,1)

        # x (batch, tokens, features) are the tokens.
        q = self.w_qs(x)
        k = self.w_ks(x)
        v = self.w_vs(x)
        attn = torch.einsum('blk,btk->blt', [q, k]) / np.sqrt(q.shape[-1])
        attn = attn.softmax(dim=-1)
        output = torch.einsum('bhlt,bhtv->bhlv', [attn, v])
        output = rearrange(output, 'b head l v -> b (head v) l')
        output = self.dropout(self.fc(output))
        return output, attn

