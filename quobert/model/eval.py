import csv
import logging
import os
from operator import itemgetter

import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm.auto import tqdm

from quobert.utils.data import collate_batch_eval

logger = logging.getLogger(__name__)


def get_most_probable_entity(proba, entities):
    most_probable_entity_idx = proba.argmax().item()
    for entity, val in entities.items():
        if most_probable_entity_idx in val[0]:
            return entity
    return "None"


def evaluate(args, model, dataset, no_save=False, has_target=True, output_proba=True):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    sampler = RandomSampler(dataset)
    loader = DataLoader(
        dataset,
        sampler=sampler,
        collate_fn=collate_batch_eval,
        batch_size=args.eval_batch_size,
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    logger.info("*** Start the evaluation porcess ***")
    logger.info(f"    Number of samples: {len(dataset)}")
    logger.info(f"    Batch size: {args.eval_batch_size} using {args.n_gpu} GPU(s)")

    correct_pos, correct_neg = 0, 0
    total_pos, total_neg = 0, 0
    model.eval()
    out = []

    for batch in tqdm(loader, desc="Evaluating"):
        batch = {
            k: v.to(args.device) if hasattr(v, "to") else v for k, v in batch.items()
        }
        with torch.no_grad():
            scores = model(
                input_ids=batch["input_ids"],
                mask_ids=batch["mask_ids"],
                attention_mask=batch["attention_mask"],
            )[0].cpu()

        for proba, entities, speaker, uid in zip(
            scores, batch["entities"], batch["speakers"], batch["uid"]
        ):
            speakers_proba = {
                entity: proba[val[0]].sum().item() for entity, val in entities.items()
            }
            speakers_proba["None"] = proba[0].item()
            speakers_proba_sorted = sorted(
                speakers_proba.items(), key=itemgetter(1), reverse=True
            )

            most_probable_speaker = speakers_proba_sorted[0][0]
            most_probable_entity = get_most_probable_entity(proba, entities)

            if has_target:
                target_speaker = speaker if speaker in entities else "None"
                is_correct = most_probable_speaker == target_speaker
                if target_speaker == "None":
                    total_neg += 1
                    if is_correct:
                        correct_neg += 1
                else:
                    total_pos += 1
                    if is_correct:
                        correct_pos += 1

            out.append(
                (
                    uid,
                    speakers_proba_sorted,
                    most_probable_speaker,
                    most_probable_entity,
                )
            )

    if has_target:
        EM_neg = correct_neg / total_neg
        EM_pos = correct_pos / total_pos
        total = total_neg + total_pos
        EM = (correct_neg + correct_pos) / (total_neg + total_pos)
        logger.info(f"EM value: {EM:.2%}%, total: {total}")
        logger.info(
            f"EM pos: {EM_pos:.2%}%, total: {total_pos} ({total_pos / total:.2%})"
        )
        logger.info(
            f"EM neg: {EM_neg:.2%}%, total: {total_neg} ({total_neg / total:.2%})"
        )

    if not no_save:
        with open(
            os.path.join(args.output_file), "w", encoding="utf-8", newline="",
        ) as csvfile:
            csvwriter = csv.writer(csvfile)
            if output_proba:
                csvwriter.writerow(
                    ["articleUID", "articleOffset", "rank", "speaker", "proba"]
                )
                for uid, speakers_proba, _, _ in out:
                    articleUID, articleOffset = uid.split()
                    for i, (speaker, proba) in enumerate(speakers_proba):
                        csvwriter.writerow(
                            [
                                articleUID,
                                articleOffset,
                                i,
                                speaker,
                                round(proba * 100, 2),
                            ]
                        )
            else:
                csvwriter.writerow(
                    ["articleUID", "articleOffset", "sum_speaker", "max_speaker"]
                )
                for uid, _, sum_speaker, max_speaker in out:
                    articleUID, articleOffset = uid.split()
                    csvwriter.writerow(
                        [articleUID, articleOffset, sum_speaker, max_speaker]
                    )
