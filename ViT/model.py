import torch
import torch.nn as nn

class LinearProjection(nn.Module):
    # 패치 리니어 프로젝션 -> 클래스토큰 붙이기 -> 포지셔널 임베딩
    def __init__(self, patch_vec_size, num_patches, latent_vec_dim, drop_rate):
        super().__init__()
        self.linear_proj = nn.Linear(patch_vec_size, latent_vec_dim)    
        # (batch_size, 나눠진패치개수, p^2*c) -> (batch_size, 나눠진패치개수, d)
        # 나눠진패치개수 = img_size / patch_size
        
        self.cls_token = nn.Parameter(torch.randn(1, latent_vec_dim))   
        # (1, d)짜리를 생성, 학습 가능해야하므로 Parameter로 정의함
        
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, latent_vec_dim)) # (1, 패치수+1, d), 얘도 학습 가능해야함
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.cat([self.cls_token.repeat(batch_size, 1, 1), self.linear_proj(x)], dim=1)
        # (1, d)짜리를 batch_size 만큼 repeat하여 맨 왼쪽에 붙임, (batch_size, 1, d)
        x += self.pos_embedding
        x = self.dropout(x)
        return x

class MultiheadedSelfAttention(nn.Module):
    def __init__(self, latent_vec_dim, num_heads, drop_rate):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_heads = num_heads
        self.latent_vec_dim = latent_vec_dim
        self.head_dim = int(latent_vec_dim / num_heads)
        self.query = nn.Linear(latent_vec_dim, latent_vec_dim)
        self.key = nn.Linear(latent_vec_dim, latent_vec_dim)
        self.value = nn.Linear(latent_vec_dim, latent_vec_dim)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)  
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        batch_size = x.size(0)
        q = self.query(x)   # (batch_size, 나눠진패치개수, d)
        k = self.key(x)
        v = self.value(x)
        
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3) 
        # (batch_size, num_heads, 나눠진패치개수, head_dim) : d를 head개수와 head별 dimesion으로 나눈 후 계산할 차원 맨 뒤로 빼서 사용
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,3,1) # k.T (batch_size, num_heads, head_dim, 나눠진패치개수)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        attention = torch.softmax(q @ k / self.scale, dim=-1)   # 루트(head차원)으로 나눠줌. 마지막차원에 대한 softmax이므로 한 개 행을 다 더하면 1됨
        x = self.dropout(attention) @ v
        x = x.permute(0,2,1,3).reshape(batch_size, -1, self.latent_vec_dim)

        return x, attention

class TFencoderLayer(nn.Module):
    def __init__(self, latent_vec_dim, num_heads, mlp_hidden_dim, drop_rate):
        super().__init__()
        self.ln1 = nn.LayerNorm(latent_vec_dim)
        self.ln2 = nn.LayerNorm(latent_vec_dim)
        self.msa = MultiheadedSelfAttention(latent_vec_dim=latent_vec_dim, num_heads=num_heads, drop_rate=drop_rate)
        self.dropout = nn.Dropout(drop_rate)
        self.mlp = nn.Sequential(nn.Linear(latent_vec_dim, mlp_hidden_dim), # Transformer의 피드포워드 역할 함
                                 nn.GELU(), nn.Dropout(drop_rate),
                                 nn.Linear(mlp_hidden_dim, latent_vec_dim),
                                 nn.Dropout(drop_rate))

    def forward(self, x):
        z = self.ln1(x)
        z, att = self.msa(z)
        z = self.dropout(z)
        x = x + z
        z = self.ln2(x)
        z = self.mlp(z)
        x = x + z

        return x, att

class VisionTransformer(nn.Module):
    def __init__(self, patch_vec_size, num_patches, latent_vec_dim, num_heads, mlp_hidden_dim, drop_rate, num_layers, num_classes):
        super().__init__()
        self.patchembedding = LinearProjection(patch_vec_size=patch_vec_size, num_patches=num_patches,
                                               latent_vec_dim=latent_vec_dim, drop_rate=drop_rate)
        # patch_vec_size = (p^2, chennel), num_patches = H*w / p^2, latent_vec_dim = 모델 풀 dimmension
        
        self.transformer = nn.ModuleList([TFencoderLayer(latent_vec_dim=latent_vec_dim, num_heads=num_heads,
                                                         mlp_hidden_dim=mlp_hidden_dim, drop_rate=drop_rate)
                                          for _ in range(num_layers)]) # 레이어 쌓기, 각 레이어는 독립적 파라미터 사용함

        self.mlp_head = nn.Sequential(nn.LayerNorm(latent_vec_dim), nn.Linear(latent_vec_dim, num_classes))
        # 클래스 토큰만(1, latent_vec_dim) 떼 와서 마지막에 output으로 클래스 개수만큼으로 출력

    def forward(self, x):
        att_list = []
        x = self.patchembedding(x)
        for layer in self.transformer:
            x, att = layer(x)
            att_list.append(att)
        x = self.mlp_head(x[:,0])   # 클래스토큰만 떼다가 mlp_head에 넣음

        return x, att_list
