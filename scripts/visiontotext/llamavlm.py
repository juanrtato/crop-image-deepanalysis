import torch
import torch.nn as nn
from unsloth import FastLanguageModel

class LlamaVLM(nn.Module):
    def __init__(self, encoder, decoder_model="unsloth/Llama-3.2-1B-Instruct", input_dim=128, decoder_dim=2048):
        super().__init__()
        self.encoder = encoder  # Encoder from TSViT
        self.decoder, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = decoder_model, # or choose "unsloth/Llama-3.2-1B-Instruct"
            max_seq_length = decoder_dim,
            dtype = None,
            load_in_4bit = True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = next(self.parameters()).device
        # self.projector = nn.Linear(input_dim, decoder_dim)  # Project visual embeddings to LLM dimension
        self.adapter = LlamaVLMAdapter(decoder_dim, input_dim)
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, image_sequence, input_ids=None, attention_mask=None, labels=None):
        device = next(self.parameters()).device
        batch_size = image_sequence.size(0)

        # Obtain visual embeddings from the encoder
        visual_emb = self.encoder.get_encoder_embeddings(image_sequence)  # [B, T_v, D_enc]

        # Extract the last token from the visual embeddings and project it
        visual_cls = visual_emb[:, 0, :]                                # [B, D_enc]
        # visual_proj = self.projector(visual_cls).unsqueeze(1)            # [B, 1, D_dec]
        visual_proj = self.adapter(visual_cls).unsqueeze(1)

        # Embeddings of the imput text (already tokenized)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else torch.ones_like(input_ids).to(device)

        # text_emb = self.decoder.transformer.wte(input_ids)              # [B, T_text, D_dec]
        text_emb = self.decoder.model.embed_tokens(input_ids)
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
    
class LlamaVLMAdapter(torch.nn.Module):
    def __init__(self, lang_embed_dim, vision_dim):
        super().__init__()
        self.activation = torch.nn.ReLU()
        self.layer1 = torch.nn.Linear(vision_dim, 500)
        self.layer2 = torch.nn.Linear(500, 500)
        self.layer3 = torch.nn.Linear(500, lang_embed_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        output = self.activation(x)
        return output
