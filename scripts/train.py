import argparse
import yaml
import torch
from pastis24 import get_dataloaders
from tsvit import torch_utils, model_architecture
from visiontotext import visiontotextmodel
import os


def training_step(model, batch, optimizer, device):
    model.train()
    sample_dict, texts, img_path = batch

    image_sequence = sample_dict['inputs'].to(device)
    tokenizer = model.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    input_ids = tokenizer.input_ids

    loss, _ = model(image_sequence, input_ids)
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for sample in dataloader:
            loss = training_step(model, sample, optimizer=None, device=device)
            total_loss += loss
    return total_loss / len(dataloader)

def main(args):
    with open(args.dataloader_config, "r") as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = get_dataloaders(config)

    encoder = model_architecture.get_model(config, device)
    encoder.load_state_dict(torch.load(args.model_path, map_location=device))
    encoder.eval()

    model_vtt = visiontotextmodel.VisionToTextModel(encoder, decoder_model="gpt2", input_dim=128).to(device)
    optimizer = torch.optim.Adam(model_vtt.parameters(), lr=1e-4)
    model_vtt.train()
    for epoch in range(args.epochs):
        print(f"[Epoch {epoch+1}/{args.epochs}]")
        epoch_loss = 0.0
        for batch in dataloader['train']:
            loss = training_step(model_vtt, batch, optimizer, device)
            epoch_loss += loss
        avg_loss = epoch_loss / len(dataloader['train'])
        print(f"Loss: {avg_loss:.4f}")

    val_loss = evaluate(model_vtt, dataloader['eval'], device)
    print(f"Eval Loss: {val_loss:.4f}")

    os.makedirs("outputs", exist_ok=True)
    torch.save(model_vtt.state_dict(), "outputs/model_vtt_gpt2.pth")
    print("✅ Modelo guardado en outputs/model_vtt_gpt2.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='../datalake/TSVIT/best.pth')
    parser.add_argument("--dataloader_config", type=str, default='../datalake/config_vtt.json')
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--param_file", type=str, help="Ruta al archivo YAML con parámetros", default=None)
    args = parser.parse_args()
    if args.param_file:
        import yaml
        with open(args.param_file, "r") as f:
            param_dict = yaml.safe_load(f)
        for k, v in param_dict.items():
            if hasattr(args, k):
                setattr(args, k, v)

    main(args)