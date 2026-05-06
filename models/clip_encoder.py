"""
CLIP Text Encoder Wrapper

Wraps OpenAI CLIP (via open_clip or transformers) to produce:
  - Token-level embeddings  (B, 77, 512) for cross-attention
  - Sentence-level embedding (B, 512) for global conditioning

Falls back gracefully if CLIP is unavailable (returns random embeddings for
smoke testing purposes).

Supported backends:
  1. open_clip  (pip install open-clip-torch)  — recommended
  2. transformers CLIP  (pip install transformers)
  3. Dummy fallback (for debugging without GPU/internet)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union


# ---------------------------------------------------------------------------
# CLIP Wrapper
# ---------------------------------------------------------------------------

class CLIPTextEncoder(nn.Module):
    """
    Encodes text strings into CLIP embeddings.

    Outputs:
        token_emb:    (B, 77, clip_dim)  — per-token for cross-attention
        sentence_emb: (B, clip_dim)      — pooled sentence embedding

    Usage:
        encoder = CLIPTextEncoder(model_name='ViT-B-32')
        token_emb, sent_emb = encoder(["a dog on grass", "a red car"])
    """

    SUPPORTED = {
        'ViT-B-32':  {'dim': 512,  'tokens': 77},
        'ViT-B-16':  {'dim': 512,  'tokens': 77},
        'ViT-L-14':  {'dim': 768,  'tokens': 77},
        'ViT-H-14':  {'dim': 1024, 'tokens': 77},
    }

    def __init__(
        self,
        model_name: str = 'ViT-B-32',
        pretrained: str = 'openai',
        device: Optional[str] = None,
        freeze: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.clip_dim = self.SUPPORTED.get(model_name, {}).get('dim', 512)
        self.max_tokens = self.SUPPORTED.get(model_name, {}).get('tokens', 77)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._backend = None

        # Try open_clip first
        try:
            import open_clip
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)
            self._backend = 'open_clip'
            print(f"CLIP backend: open_clip ({model_name}, {pretrained})")
        except ImportError:
            pass

        # Try transformers
        if self._backend is None:
            try:
                from transformers import CLIPTokenizer, CLIPTextModel
                hf_name = f'openai/clip-{model_name.lower().replace("-", "-")}'
                self.tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
                self.clip_model = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32')
                self._backend = 'transformers'
                print(f"CLIP backend: transformers (openai/clip-vit-base-patch32)")
            except ImportError:
                pass

        if self._backend is None:
            print("WARNING: CLIP not available. Using dummy encoder (for debugging only).")
            print("Install: pip install open-clip-torch")
            self._backend = 'dummy'
            # Learnable projection for dummy mode
            self.dummy_proj = nn.Linear(1, self.clip_dim)

        if self._backend != 'dummy':
            self.clip_model = self.clip_model.to(self.device)
            if freeze:
                for p in self.clip_model.parameters():
                    p.requires_grad_(False)

    @property
    def embed_dim(self) -> int:
        return self.clip_dim

    @torch.no_grad()
    def encode_text(
        self, texts: Union[List[str], str]
    ) -> tuple:
        """
        Encode a list of text strings.
        Returns (token_embeddings, sentence_embeddings).
        """
        if isinstance(texts, str):
            texts = [texts]

        B = len(texts)

        if self._backend == 'dummy':
            token_emb = torch.randn(B, self.max_tokens, self.clip_dim)
            sent_emb  = F.normalize(torch.randn(B, self.clip_dim), dim=-1)
            return token_emb, sent_emb

        if self._backend == 'open_clip':
            import open_clip
            tokens = self.tokenizer(texts).to(self.device)
            # open_clip encode_text returns sentence embedding
            # We need intermediate features for token-level embeddings
            sent_emb = self.clip_model.encode_text(tokens)
            sent_emb = F.normalize(sent_emb.float(), dim=-1)

            # Get token-level features via forward hook
            token_emb = self._get_token_features_open_clip(tokens, B)
            return token_emb.float(), sent_emb.float()

        if self._backend == 'transformers':
            enc = self.tokenizer(
                texts, return_tensors='pt', padding='max_length',
                truncation=True, max_length=self.max_tokens
            ).to(self.device)
            out = self.clip_model(**enc)
            token_emb = out.last_hidden_state.float()       # (B, 77, dim)
            sent_emb  = F.normalize(out.pooler_output.float(), dim=-1)
            return token_emb, sent_emb

    def _get_token_features_open_clip(self, tokens, B):
        """Extract per-token features from open_clip via manual forward."""
        try:
            model = self.clip_model
            x = model.token_embedding(tokens)
            x = x + model.positional_embedding
            x = x.permute(1, 0, 2)   # NLD -> LND
            x = model.transformer(x)
            x = x.permute(1, 0, 2)   # LND -> NLD
            x = model.ln_final(x)
            return x.float()
        except Exception:
            # Fallback: repeat sentence embedding
            sent = self.clip_model.encode_text(tokens).float()
            return sent.unsqueeze(1).expand(-1, self.max_tokens, -1)

    def forward(self, texts: Union[List[str], str]):
        return self.encode_text(texts)


# ---------------------------------------------------------------------------
# CLIP Conditioning Projector
# ---------------------------------------------------------------------------

class CLIPProjector(nn.Module):
    """
    Optional learned projection layer to adapt CLIP embeddings to generator's
    expected clip_dim if they differ.
    Also adds learned positional embeddings to token sequence.
    """

    def __init__(self, clip_dim: int, target_dim: int, num_tokens: int = 77):
        super().__init__()
        self.proj = nn.Linear(clip_dim, target_dim) if clip_dim != target_dim else nn.Identity()
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, target_dim) * 0.02)

    def forward(self, token_emb: torch.Tensor) -> torch.Tensor:
        x = self.proj(token_emb)
        return x + self.pos_emb[:, :x.shape[1], :]


# ---------------------------------------------------------------------------
# Null/Empty text embedding (for unconditional generation)
# ---------------------------------------------------------------------------

class NullTextEmbedding(nn.Module):
    """
    Learned null embedding used when no text is provided.
    Analogous to the null class token in CFG for diffusion.
    """

    def __init__(self, clip_dim: int, num_tokens: int = 77):
        super().__init__()
        self.null_token = nn.Parameter(torch.randn(1, num_tokens, clip_dim) * 0.02)
        self.null_sent  = nn.Parameter(torch.randn(1, clip_dim) * 0.02)

    def forward(self, B: int, device: torch.device):
        return (
            self.null_token.expand(B, -1, -1).to(device),
            self.null_sent.expand(B, -1).to(device),
        )


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    encoder = CLIPTextEncoder(model_name='ViT-B-32')
    texts = ["a dog running on grass", "a red sports car on a highway"]
    token_emb, sent_emb = encoder.encode_text(texts)
    print(f"Token embeddings: {token_emb.shape}")   # (2, 77, 512)
    print(f"Sentence embeddings: {sent_emb.shape}") # (2, 512)

    proj = CLIPProjector(clip_dim=512, target_dim=512)
    projected = proj(token_emb)
    print(f"Projected: {projected.shape}")
