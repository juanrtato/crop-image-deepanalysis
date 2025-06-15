import argparse
import json
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

def evaluation_step(model, batch, device):
    model.eval()
    sample_dict, texts, img_path = batch

    image_sequence = sample_dict['inputs'].to(device)
    tokenizer = model.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    input_ids = tokenizer.input_ids

    with torch.no_grad():
        loss, _ = model(image_sequence, input_ids)
    return loss.item()

def main(args):
    with open(args.dataloader_config, "r") as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Config dates format train: {config.get('DATASETS', {}).get('train', {}).get('format_dates')}")
    print(f"Config dates format eval: {config.get('DATASETS', {}).get('eval', {}).get('format_dates')}")
    dataloader = get_dataloaders(config)

    encoder = model_architecture.get_model(config, device)
    encoder.load_state_dict(torch.load(args.model_path, map_location=device))
    encoder.eval()

    model_vtt = visiontotextmodel.VisionToTextModel(encoder, decoder_model="gpt2", input_dim=128).to(device)
    optimizer = torch.optim.Adam(model_vtt.parameters(), lr=1e-4)
    model_vtt.train()
    train_losses = []
    eval_losses = []
    os.makedirs("outputs", exist_ok=True)
    for epoch in range(args.epochs):
        print(f"[Epoch {epoch+1}/{args.epochs}]")
        train_loss = 0.0
        for batch in dataloader['train']:
            loss = training_step(model_vtt, batch, optimizer, device)
            train_loss += loss
        avg_train_loss = train_loss / len(dataloader['train'])
        print(f"Train Loss: {avg_train_loss:.4f}")
        train_losses.append(avg_train_loss)

        eval_loss = 0.0
        for batch in dataloader['eval']:
            loss = evaluation_step(model_vtt, batch, device)
            eval_loss += loss
        avg_eval_loss = eval_loss / len(dataloader['eval'])
        print(f"Eval Loss: {avg_eval_loss:.4f}")
        eval_losses.append(avg_eval_loss)

        torch.save(
            model_vtt.state_dict(),
            f"outputs/model_vtt_gpt2_epoch_{epoch+1}.pth"
        )
        print(f"✅ Modelo guardado en outputs/model_vtt_gpt2_epoch_{epoch+1}.pth")
    
    torch.save(model_vtt.state_dict(), "outputs/model_vtt_gpt2.pth")
    print("✅ Modelo guardado en outputs/model_vtt_gpt2.pth")

    with open("outputs/losses_model_vtt_gpt2.json", "w") as f:
        json.dump(
            {
                "train_loss": train_losses,
                "eval_loss": eval_losses
            }, f
        )


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