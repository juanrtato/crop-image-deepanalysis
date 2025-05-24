import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class VisionToTextModel(nn.Module):
    def __init__(self, encoder, decoder_model="gpt2", input_dim=128):
        super().__init__()
        self.encoder = encoder  # Encoder for visual embeddings
        self.decoder = GPT2LMHeadModel.from_pretrained(decoder_model)
        self.tokenizer = GPT2Tokenizer.from_pretrained(decoder_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = next(self.parameters()).device
        print(f"Decoder input size: {self.decoder.config.n_embd}") # 768
        # Projection
        self.projector = nn.Linear(input_dim, self.decoder.config.n_embd)  # Project visual embeddings to LLM dimension
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, image_sequence, input_ids=None, attention_mask=None, labels=None):
        device = next(self.parameters()).device
        batch_size = image_sequence.size(0)

        # Obtain visual embeddings from the encoder
        visual_emb = self.encoder.get_encoder_embeddings(image_sequence)  # [B, T_v, D_enc]

        # Extract the last token from the visual embeddings and project it
        visual_cls = visual_emb[:, 0, :]                                # [B, D_enc]
        visual_proj = self.projector(visual_cls).unsqueeze(1)            # [B, 1, D_dec]

        # Embeddings of the imput text (already tokenized)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else torch.ones_like(input_ids).to(device)

        text_emb = self.decoder.transformer.wte(input_ids)              # [B, T_text, D_dec]
        #print("image_sequence.shape:", image_sequence.shape)
        #print("visual_emb.shape:", visual_emb.shape)
        print(f"visual_proj: {visual_proj.shape}, text_emb: {text_emb.shape}")
        # Concatenate visual projection with text embeddings
        combined_embeddings = torch.cat([visual_proj, text_emb], dim=1)  # [B, 1 + T_text, D_dec]

        # Construct attention mask (combined)
        visual_attention_mask = torch.ones((batch_size, 1), dtype=torch.long).to(device)
        combined_attention_mask = torch.cat([visual_attention_mask, attention_mask], dim=1)

        # Create labels for the decoder
        if labels is not None:
            labels = labels.to(device)
            labels = torch.cat([torch.full((batch_size, 1), -100).to(device), labels], dim=1)  # Ignorar token visual
        else:
            labels = torch.cat([torch.full((batch_size, 1), -100).to(device), input_ids], dim=1)

        # Pasar al decoder
        outputs = self.decoder(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            labels=labels
        )

        return outputs.loss, outputs.logits
    
class Adapter(nn.Module):
    def __init__(self, input_dim=128, bottleneck_dim=64):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, input_dim)
        )

    def forward(self, x):
        return x + self.adapter(x)  # Residual connection
