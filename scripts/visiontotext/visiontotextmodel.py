from mappings import INITIAL_MSGS_EN
import torch
import torch.nn as nn
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tsvit import torch_utils, model_architecture

class VisionToTextModel(nn.Module):
    def __init__(self, encoder, decoder_model="gpt2", input_dim=128):
        super().__init__()
        self.encoder = encoder  # Encoder for visual embeddings
        self.decoder = GPT2LMHeadModel.from_pretrained(decoder_model)
        self.tokenizer = GPT2Tokenizer.from_pretrained(decoder_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = next(self.parameters()).device
        self.projector = nn.Linear(input_dim, self.decoder.config.n_embd)
        # Projection
        #self.projector = nn.Sequential(
        #    nn.Dropout(0.1),  # Dropout for regularization
        #    nn.Linear(input_dim, self.decoder.config.n_embd)  # Project visual embeddings to LLM dimension
        #)
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

        # Pass through the decoder
        outputs = self.decoder(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            labels=labels
        )

        return outputs.loss, outputs.logits

    def generate(self, input_single, prompt="The image shows", max_length=120):
        self.eval()
        self.to(input_single.device)  # <- esta línea es clave
        initial_msgs = INITIAL_MSGS_EN 
        prompt = "Description of the agricultural activity seen in the image:"
        #prompt = initial_msgs[np.random.randint(0, len(initial_msgs))].format(crop="").strip()
        
        with torch.no_grad():
            visual_emb = self.encoder.get_encoder_embeddings(input_single)
            visual_cls = visual_emb[:, 0, :]
            visual_proj = self.projector(visual_cls).unsqueeze(1)

            input_ids = self.tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(input_single.device)
            input_ids = input_ids.expand(visual_cls.shape[0], -1)
            text_emb = self.decoder.transformer.wte(input_ids)

            input_emb = torch.cat([visual_proj, text_emb], dim=1)

            visual_attention_mask = torch.ones(visual_proj.size()[:-1], dtype=torch.long).to(input_single.device)
            text_attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            attention_mask = torch.cat([visual_attention_mask, text_attention_mask], dim=1)

            pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

            outputs = self.decoder.generate(
                inputs_embeds=input_emb,
                attention_mask=attention_mask,
                max_length=max_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=1.0,
                num_return_sequences=1,
                pad_token_id=pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        return prompt + self.tokenizer.decode(outputs[0], skip_special_tokens=True)


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


def get_visiontotext_model(config, device, encoder_path, decoder_model="gpt2", input_dim=128):
    """
    Factory function to create a VisionToTextModel with an encoder and a decoder.
    """
    encoder = model_architecture.get_model(config, device)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.eval()
    model = VisionToTextModel(encoder, decoder_model, input_dim)
    return model