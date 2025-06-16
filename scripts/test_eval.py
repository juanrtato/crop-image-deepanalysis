import csv
import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from bert_score import score as bert_score
# Supone que ya adaptaste model.generate(...) como método del modelo
def generate_caption(model, image_tensor, device, max_length=200):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)
    return model.generate(image_tensor, max_length=max_length)


#def compute_bert_meteor_score(model, dataloader, device):
def compute_metrics(model, dataloader, device, save_path=None):
    model.eval()
    meteor_scores = []
    bleu_scores = []
    references_all = []
    predictions_all = []
    results = []

    smooth_fn = SmoothingFunction().method4

    for batch in dataloader:
        sample_dict, references, img_paths = batch
        image_seq = sample_dict['inputs'].to(device)

        for i in range(len(references)):
            ref_text = "Description of the agricultural activity seen in the image: " + references[i]
            pred_text = generate_caption(model, image_seq[i], device)

            # Tokenización
            ref_tokens = nltk.word_tokenize(ref_text.lower())
            pred_tokens = nltk.word_tokenize(pred_text.lower())

            # Métricas individuales
            meteor = single_meteor_score(ref_tokens, pred_tokens)
            bleu = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth_fn)

            meteor_scores.append(meteor)
            bleu_scores.append(bleu)

            references_all.append(ref_text)
            predictions_all.append(pred_text)

            results.append({
                "img_path": img_paths[i],
                "ground_truth": ref_text,
                "prediction": pred_text,
                "meteor": meteor,
                "bleu": bleu
            })

            print(f"Ref:  {ref_text}")
            print(f"Pred: {pred_text}")
            print(f"METEOR: {meteor:.4f} | BLEU: {bleu:.4f}")
            print("------")

    # BERTScore (por lote)
    _, _, F1 = bert_score(predictions_all, references_all, lang="en", rescale_with_baseline=True)
    bert_avg = F1.mean().item()
    meteor_avg = sum(meteor_scores) / len(meteor_scores)
    bleu_avg = sum(bleu_scores) / len(bleu_scores)

    # Añadir BERT individual
    for idx in range(len(results)):
        results[idx]["bert"] = F1[idx].item()

    # Score combinado 
    combined_score = 0.5 * bert_avg + 0.5 * meteor_avg

    print(f"\n🤖 BERTScore promedio: {bert_avg:.4f}")
    print(f"🌾 METEOR promedio:    {meteor_avg:.4f}")
    print(f"📘 BLEU promedio:      {bleu_avg:.4f}")
    print(f"🧪 Score combinado:    {combined_score:.4f}")

    return {
        "bert": bert_avg,
        "meteor": meteor_avg,
        "bleu": bleu_avg,
        "combined": combined_score,
        "per_sample": results
    }
