import torch
import torch.nn as nn
import torch.nn.functional as F
from util import sample_and_group 
from einops.layers.torch import Rearrange
from einops import reduce, rearrange, repeat
import numpy as np
from point2vec.pointnet import PointcloudTokenizer

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
    def __init__(self, args, output_channels):
        super(Pct, self).__init__()
        self.args = args
        self.n_tokens = args.train.n_tokens
        self.conv1 = nn.Conv1d(3, 192, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(192, 256, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(192)
        self.bn2 = nn.BatchNorm1d(256)
        self.gather_local_0 = Local_op(in_channels=6, out_channels=64)
        self.gather_local_1 = Local_op(in_channels=128, out_channels=256)
        #self.gather_local_1 = Local_op(in_channels=6, out_channels=self.n_tokens)

        if args.train.adaptive.is_adaptive:   
            self.pt_last = Point_Transformer_Adaptive(args, channels=256, d_model=512, d_k=32, d_v=64, n_heads=8, n_blocks=args.train.n_blocks, layers_to_drop=args.train.adaptive.layers_to_drop)
        elif args.train.merger.is_merger:
            self.pt_last = Point_Transformer_Merger(args, channels=256, d_model=512, d_k=32, d_v=64, n_heads=8, n_blocks=args.train.n_blocks)
        else:
            self.pt_last = Point_Transformer_Last(args, channels=256, d_model=512, d_k=32, d_v=64, n_heads=8, n_blocks=args.train.n_blocks)
        



        self.linear1 = nn.Linear(512, 256, bias=False)
        self.bn6 = nn.BatchNorm1d(256)
        self.dp1 = nn.Dropout(p=args.train.dropout)
        self.linear2 = nn.Linear(256, 128)
        self.bn7 = nn.BatchNorm1d(128)
        self.dp2 = nn.Dropout(p=args.train.dropout)
        self.linear3 = nn.Linear(128, output_channels)

        self.positional_encoding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, 256),
        )
        self.tokenizer = PointcloudTokenizer(
            num_groups=256,
            group_size=32,
            group_radius=None,
            token_dim=256,
        )


    def forward(self, x, drop_temp=1):
        #OLD PATCHING METHOD
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        # B, D, N
        #x = F.relu(self.bn1(self.conv1(x)))
        # B, D, N
        #x = F.relu(self.bn2(self.conv2(x)))
        
        #print(x.shape)

        #x = F.relu(self.bn1(self.conv1(x)))
        #feature_1 = F.relu(self.bn2(self.conv2(x)))

        #print(feature_1.shape)

        #patchifier
        
        x = x.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)         
        feature_0 = self.gather_local_0(new_feature)
        
        #print("feature0:", feature_0.shape)
        
        feature = feature_0.permute(0, 2, 1)
        new_xyz, new_feature = sample_and_group(npoint=self.n_tokens, radius=0.2, nsample=32, xyz=new_xyz, points=feature) 
        feature_1 = self.gather_local_1(new_feature)
        
        
        
        #print("feature1:", feature_1.shape)
        
        #print(x.shape)
        """
        tokens, centers = self.tokenizer(x.permute(0,2,1))

        pos_embedding=False
        if pos_embedding:
            self.cls_pos = nn.Parameter(torch.zeros(256))
            pos_embeddings = self.positional_encoding(centers)

        feature_1 = tokens
        """

        if self.args.train.adaptive.is_adaptive:
            x, masks,distr = self.pt_last(feature_1, drop_temp=drop_temp)
        elif self.args.train.merger.is_merger:
            x, merge_ref= self.pt_last(feature_1)
        else:
            x = self.pt_last(feature_1)
        #x, masks = self.pt_last(x)
        tokens = x
        #print("output SA:", x.shape)

        #POOLING INSTEAD OF CLASS TOKEN
        #x = F.adaptive_max_pool1d(x[:,:-1,:], 2).view(batch_size, -1)
        #x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)

        if self.args.task == "classification":
            #CLASS TOKEN
            x = F.leaky_relu(self.bn6(self.linear1(x[:,-1,:])), negative_slope=0.2)
            

            x = self.dp1(x)
            x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
            x = self.dp2(x)
            x = self.linear3(x)
            if self.args.train.adaptive.is_adaptive:
                if self.args.visualize_pc:
                    return x, masks, distr, tokens, xyz, new_xyz
                else:
                    return x, masks, distr, tokens
            elif self.args.train.merger.is_merger:
                if self.args.visualize_pc:
                    return x, xyz, new_xyz
                else:
                    return x
            else:
                return x, tokens
        
        elif self.args.task == "segmentation":

            #DISCARD CLASS TOKEN
            x = F.leaky_relu(self.bn6(self.linear1(x[:,:-1,:])), negative_slope=0.2)
            x = self.dp1(x)
            x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
            x = self.dp2(x)
            x = self.linear3(x)
            if self.args.visualize_pc:
                return x, merge_ref, xyz, new_xyz
            else:
                return x, merge_ref



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
  

def batch_index_select(x, idx):
    if len(x.size()) == 3:
        B, N, C = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N, C)[idx.reshape(-1)].reshape(B, N_new, C)
        return out
    elif len(x.size()) == 2:
        B, N = x.size()
        N_new = idx.size(1)
        offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
        idx = idx + offset
        out = x.reshape(B*N)[idx.reshape(-1)].reshape(B, N_new)
        return out
    else:
        raise NotImplementedError


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
        x = torch.cat([repeat(self.cls_token, 'v -> b 1 v', b=x.shape[0]), x], dim=1)
    
        # Initialize drop decisions
        B, P, _ = x.shape
        prev_decision = torch.ones(B, P-1, 1, dtype=x.dtype, device=x.device)
        policy = torch.ones(B, P, 1, dtype=x.dtype, device=x.device)
        #policy = torch.zeros(B, P, 1, dtype=x.dtype, device=x.device)
        out_pred_prob = []
        pred_distr = [[],[],[],[]]
        p = 0

        for i, l in enumerate(self.blocks):
            if i in self.layers_to_drop:
                # Ignore the class token
                points_x = x[:, 1:]
                # Current drop score
                pred_score = self.score_predictor[p](points_x, prev_decision)#.reshape(B, -1, 2)
                #for visualization purposes
                pred_distr[p].append(pred_score.reshape(-1,2))
                # Slow warmup
                keepall = torch.cat((torch.zeros_like(pred_score[:,:,0:1]), torch.ones_like(pred_score[:,:,1:2])),2) 
                pred_score = pred_score*drop_temp + keepall*(1-drop_temp)
                
                if self.training:
                    # Convert to log-prob
                    pred_score = torch.log(pred_score + 1e-8)
                    # Sample mask and update previous one
                    hard_keep_decision = F.gumbel_softmax(pred_score, hard=True)[:, :, 1:2] * prev_decision
                    # OR Deterministic mask and update previous one
                    #soft = F.softmax(pred_score, dim=-1)[:, :, 1:2]
                    #hard = torch.argmax(pred_score, dim=-1).float().unsqueeze(-1)
                    #decision = soft + (hard - soft).detach()
                    #hard_keep_decision = decision * prev_decision
                    
                    #for visualization purposes
                    out_pred_prob.append(hard_keep_decision.reshape(B, P-1))
                    # Build the full policy (always keep the class token)
                    cls_policy = torch.ones(B, 1, 1, dtype=hard_keep_decision.dtype, device=hard_keep_decision.device)
                    policy = torch.cat([cls_policy, hard_keep_decision], dim=1)
                    prev_decision = hard_keep_decision
                    x = l(x, policy=policy)
                else:
                    # Treshold mask and update previous one
                    #hard_keep_decision = (pred_score[:, :, 1:2] > 0.9).float() * prev_decision
                    # OR Deterministic mask with fixed number of tokens kept and update previous one
                    score = pred_score[:,:,1]
                    num_keep_tokens = int((1-self.args.train.adaptive.drop_ratio[p]) * (P-1))
                    keep_policy = torch.argsort(score, dim=1, descending=True)[:, :num_keep_tokens]
                    cls_policy = torch.zeros(B, 1, dtype=keep_policy.dtype, device=keep_policy.device)
                    now_policy = torch.cat([cls_policy, keep_policy+1], dim=1)
                    x = batch_index_select(x, now_policy)
                    prev_decision = batch_index_select(prev_decision, keep_policy)
                    x = l(x)
                    #print(i, x.shape)
                p += 1


                #"Dropout", added for debugging purposes"
                #stuff = (1-self.args.drop_ratio[i])*drop_temp + (1-drop_temp)
                #hard_keep_decision = torch.bernoulli(torch.ones_like(hard_keep_decision)*stuff).float().to(hard_keep_decision.device)

            else:
                if self.training:
                    x = l(x, policy=policy)   
                else:
                    x = l(x)
                

        return x, out_pred_prob, pred_distr



class CrossAttention(nn.Module):
    
    def __init__(self, d_model, d_k, d_v, n_q, dropout=0.1):
        super().__init__()

        self.w_ks = nn.Conv1d(d_model, d_k, kernel_size=1, bias=False)
        self.w_vs = nn.Conv1d(d_model, d_v, kernel_size=1, bias=False)
        self.q = nn.Parameter(torch.randn((n_q,d_k), requires_grad=True)) # trainable queries
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.fc = nn.Conv1d(d_v, d_model, kernel_size=1, bias=False)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x, ):

        x = x.permute(0,2,1)

        # x (batch, tokens, features) are the tokens.
        q = repeat(self.q, 'nq dk -> b dk nq', b=x.shape[0])
        k = self.w_ks(x)
        v = self.w_vs(x)
        #v = self.v
        #print(q.shape, k.shape)
        attn = torch.einsum('bkt,bkl->blt', [q, k]) / np.sqrt(q.shape[-1])
        
        
        if self.training: 
            #ATTENTION WITH GUMBEL
            hardattn = F.gumbel_softmax(attn, hard=True)
        else:
            #HARD ATTENTION WITH ARGMAX
            hard = torch.argmax(attn, dim=-1)
            hardattn = torch.zeros_like(attn).scatter_(-1, hard.unsqueeze(-1), 1)
        
        #print(attn.shape, v.shape)
        output = torch.einsum('btl,bvt->blv', [hardattn, v])
        output = rearrange(output, 'b l v -> b v l')
        output = self.dropout(self.fc(output)).permute(0,2,1)
        return output, hardattn
    

class CrossTransformerBlock(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_q, dropout=0.1):
        super().__init__()
        self.sa = CrossAttention(d_model, d_k, d_v, n_q=n_q,dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.GELU(),
            nn.Linear(d_model*2, d_model)
        )

    def forward(self, x):
        x, hardattn = self.sa(self.ln1(x))
        x = self.ff(self.ln2(x))
        return x, hardattn
    


class Point_Transformer_Merger(nn.Module):
    def __init__(self, args, channels=256, d_model = 256, d_k=16, d_v=32, n_heads=8, n_blocks=4):
        super(Point_Transformer_Merger, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(channels, d_model, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(d_model)
        self.bn2 = nn.BatchNorm1d(d_model)

        self.cls_token = nn.Parameter(torch.randn((d_model,), requires_grad=True)) # Class token
        self.blocks = nn.ModuleList([TransformerBlock(d_model, d_k, d_v, n_heads) for _ in range(n_blocks)]) # Transformer blocks
        self.mergers = nn.ModuleList([CrossTransformerBlock(d_model, d_k, d_v, args.train.merger.n_q[i]) for i in range(n_blocks)]) # Transformer blocks
        #self.layers_to_drop = layers_to_drop
        #self.score_predictor = nn.ModuleList([DropPredictor(d_model) for _ in range(n_blocks)])

    def forward(self, x):
        
        merge_ref = torch.arange(0, x.shape[2]).repeat(x.shape[0], 1).to(x.device)
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = x.permute(0,2,1)
        x = torch.cat([x, repeat(self.cls_token, 'v -> b 1 v', b=x.shape[0])], dim=1)
    
        for i, l in enumerate(self.blocks):

            points_x = x[:, :-1]                
            points_x, hardattn = self.mergers[i](points_x)

            assign = torch.argmax(hardattn, dim=-1)
            newref = torch.gather(assign, dim=1, index=merge_ref)
            merge_ref = newref

            x = torch.cat([points_x, x[:, -1:]], dim=1)
            x = l(x)

        return x, merge_ref
    

class Pct_nogroup(nn.Module):
    def __init__(self, args, output_channels=86):
        super(Pct_nogroup, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        #self.gather_local_0 = Local_op(in_channels=128, out_channels=128)
        #self.gather_local_1 = Local_op(in_channels=256, out_channels=256)

        if args.train.adaptive.is_adaptive:   
            self.pt_last = Point_Transformer_Adaptive(args, channels=128, d_model=512, d_k=32, d_v=64, n_heads=8, n_blocks=4, layers_to_drop=args.train.adaptive.layers_to_drop)
        elif args.train.merger.is_merger:
            self.pt_last = Point_Transformer_Merger(args, channels=128, d_model=512, d_k=32, d_v=64, n_heads=8, n_blocks=4)
        else:
            self.pt_last = Point_Transformer_Last(args, channels=128, d_model=512, d_k=32, d_v=64, n_heads=8, n_blocks=4)
        


        self.conv3 = nn.Conv1d(512, 256, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(256, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, output_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(128)
        self.dp1 = nn.Dropout(p=args.train.dropout)
        self.dp2 = nn.Dropout(p=args.train.dropout)


    def forward(self, x, drop_temp=1):
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        # B, D, N
        x = F.relu(self.bn1(self.conv1(x)))
        # B, D, N
        x = F.relu(self.bn2(self.conv2(x)))
        
        #x = x.permute(0, 2, 1)
        #new_xyz, new_feature = sample_and_group(npoint=512, radius=0.15, nsample=32, xyz=xyz, points=x)         
        #feature_0 = self.gather_local_0(new_feature)
        
        #print("feature0:", feature_0.shape)
        
        #feature = feature_0.permute(0, 2, 1)
        #new_xyz, new_feature = sample_and_group(npoint=256, radius=0.2, nsample=32, xyz=new_xyz, points=feature) 
        #feature_1 = self.gather_local_1(new_feature)

        #print("feature1:", feature_1.shape)

        if self.args.train.adaptive.is_adaptive:
            x, masks,distr = self.pt_last(x, drop_temp=drop_temp)
        elif self.args.train.merger.is_merger:
            x, merge_ref = self.pt_last(x)
        else:
            x = self.pt_last(x)
        #x, masks = self.pt_last(x)

        #print("output SA:", x.shape)

        #POOLING INSTEAD OF CLASS TOKEN
        #x = F.adaptive_max_pool1d(x[:,:-1,:], 2).view(batch_size, -1)
        #x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)

        #CLASS TOKEN
        #x = F.leaky_relu(self.bn6(self.linear1(x[:,-1,:])), negative_slope=0.2)
        
        #SEGMENTATION
        x = x[:,:-1,:].permute(0, 2, 1) #remove cls token and reshape
        print("output transformer:", x.shape)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.conv5(x)
        if self.args.train.adaptive.is_adaptive:
            return x, masks, distr
        elif self.args.train.merger.is_merger:
            return x, merge_ref