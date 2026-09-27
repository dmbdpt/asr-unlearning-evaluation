from collections import defaultdict
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics.text import WordErrorRate, CharErrorRate

from src.utils.utils import rank0_print
from src.data.datasets.librispeech import collate_fn


def _extract_text(res) -> str:
    if isinstance(res, (list, tuple)):
        return _extract_text(res[0]) if res else ""
    return res


def _organize_by_speaker(batch_res: dict) -> dict:
    organized: dict = defaultdict(lambda: {"transcripts": [], "ground_truths": [], "speaker_ids": []})
    for i in range(len(batch_res["transcripts"])):
        spk = batch_res["speaker_ids"][i]
        organized[spk]["transcripts"].append(batch_res["transcripts"][i])
        organized[spk]["ground_truths"].append(batch_res["ground_truths"][i])
        organized[spk]["speaker_ids"].append(spk)
    return organized


def run_evaluate_metrics(model, datasets, sets_to_evaluate=["test"], cfg=None):
    if cfg is None:
        cfg = {}

    results = {}
    model.eval()

    with torch.inference_mode():
        for set_name in sets_to_evaluate:
            rank0_print(f"[Evaluation] Evaluating {set_name} set with {len(datasets[set_name])} samples")

            batch_res = {"transcripts": [], "ground_truths": [], "speaker_ids": []}
            
            if dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                device = torch.device(f"cuda:{rank}")

                sampler = DistributedSampler(
                    datasets[set_name],
                    num_replicas=world_size,
                    rank=rank,
                    shuffle=False,
                )
                rank0_print(f"[Evaluation] Using Distributed Sampler, world size: {world_size}")
            else:
                sampler = None
                rank = 0
                world_size = 1
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print("[Evaluation] Using Single-GPU")

            data_loader = DataLoader(
                datasets[set_name],
                batch_size=1,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=cfg.num_workers,
                pin_memory=True,
                sampler=sampler,
            )

            desc = f"[RANK: {rank}] Metrics Eval: {set_name}"

            for batch in tqdm(data_loader, desc=desc, position=rank):
                wav = batch["wav"]
                wav = wav.to(device)

                raw_transcripts = model.inference(wav)
                transcripts = [_extract_text(t) for t in raw_transcripts]
                
                batch_res["transcripts"].extend(transcripts)
                batch_res["ground_truths"].extend(batch["wrd"])
                batch_res["speaker_ids"].extend(batch["speaker_id"])

            if dist.is_initialized():
                all_ranks_res = [None] * world_size
                dist.all_gather_object(all_ranks_res, batch_res)
                
                combined_res = {"transcripts": [], "ground_truths": [], "speaker_ids": []}
                for rank_data in all_ranks_res:
                    for key in combined_res:
                        combined_res[key].extend(rank_data[key])
                batch_res = combined_res

            organized_res = _organize_by_speaker(batch_res)
            results[set_name] = calculate_metrics(organized_res, cfg)

    return results

def calculate_metrics(organized_res, cfg):
    wer_calc = WordErrorRate()
    cer_calc = CharErrorRate()
    for spk in organized_res:
        all_transcripts = organized_res[spk]["transcripts"]
        all_ground_truths = organized_res[spk]["ground_truths"]

        wer = float(wer_calc(all_transcripts, all_ground_truths))
        cer = float(cer_calc(all_transcripts, all_ground_truths))

        organized_res[spk]["wer"] = wer
        organized_res[spk]["cer"] = cer

        # Per-utterance WER/CER (each utterance scored against only its own reference), as...
        organized_res[spk]["per_utt_wer"] = [
            float(wer_calc([t], [g])) for t, g in zip(all_transcripts, all_ground_truths)
        ]
        organized_res[spk]["per_utt_cer"] = [
            float(cer_calc([t], [g])) for t, g in zip(all_transcripts, all_ground_truths)
        ]

    return organized_res