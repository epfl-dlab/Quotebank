import logging
import os

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from tqdm.auto import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup

from quobert.utils import set_seed
from quobert.utils.data import collate_batch_train

logger = logging.getLogger(__name__)


def fit(
    args,
    train_dataset: Dataset,
    model,
):
    if not args.azure:
        from torch.utils.tensorboard import SummaryWriter

        logger.info("Starting the Tensorboard summary writer")
        tb_writer = SummaryWriter(args.tb_path)
    else:
        from azureml.core.run import Run

        run = Run.get_context()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    sampler = RandomSampler(train_dataset)
    loader = DataLoader(
        train_dataset,
        sampler=sampler,
        collate_fn=collate_batch_train,
        batch_size=args.train_batch_size,
    )

    t_total = (
        len(train_dataset)
        // (args.train_batch_size * args.gradient_accumulation_steps)
        * args.num_train_epochs
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(
        os.path.join(args.model_name_or_path, "optimizer.pt")
    ) and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "optimizer.pt"))
        )
        scheduler.load_state_dict(
            torch.load(os.path.join(args.model_name_or_path, "scheduler.pt"))
        )

    if args.fp16:
        try:
            from apex import amp  # type: ignore
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )

        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("*** Start the training porcess ***")
    logger.info(f"    Number of samples: {len(train_dataset)}")
    logger.info(f"    Number of epochs: {args.num_train_epochs}")
    logger.info(f"    Batch size: {args.train_batch_size} using {args.n_gpu} GPU(s)")
    logger.info(f"    Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"    Number of optimization steps: {t_total}")

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path, model is already reloaded
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained, steps_trained_in_current_epoch = divmod(
                global_step, len(loader) // args.gradient_accumulation_steps
            )

            logger.info(
                "    Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info(f"    Continuing training from epoch {epochs_trained}")
            logger.info(f"    Continuing training from global step {global_step}")
            logger.info(
                f"    Will skip the first {steps_trained_in_current_epoch} steps in the first epoch"
            )
        except ValueError:
            logger.info("    Starting fine-tuning.")

    train_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    model.train()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Current epoch"
    )
    # Added here for reproductibility
    set_seed(args.seed)

    for current_epoch in train_iterator:
        epoch_iterator = tqdm(loader, desc="Current iteration")
        current_epoch_EM = 0
        for step, batch in enumerate(epoch_iterator):
            # Skip already trained steps
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            batch = {k: v.to(args.device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                mask_ids=batch["mask_ids"],
                attention_mask=batch["attention_mask"],
                targets=batch["targets"],
            )
            loss = outputs[0]
            current_epoch_EM += outputs[1].sum().item()
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            train_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if not args.azure:
                        tb_writer.add_scalar(
                            "lr", scheduler.get_last_lr()[0], global_step
                        )
                        tb_writer.add_scalar(
                            "Loss/train",
                            (train_loss - logging_loss) / args.logging_steps,
                            global_step,
                        )
                    else:
                        run.log_row(
                            "lr", lr=scheduler.get_last_lr()[0], step=global_step
                        )
                        run.log_row(
                            "Loss/train",
                            loss=(train_loss - logging_loss) / args.logging_steps,
                            step=global_step,
                        )
                    logging_loss = train_loss

                # Save model checkpoint
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(
                        args.output_dir, f"checkpoint-{global_step:05d}"
                    )
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # Handle torch.nn.DataParallel
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info(f"Saved model checkpoint to {output_dir}")

                    torch.save(
                        optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                    )
                    torch.save(
                        scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                    )
                    logger.info(f"Saved optimizer and scheduler states to {output_dir}")
        if not args.azure:
            tb_writer.add_scalar(
                "EM/train", current_epoch_EM * 100 / len(train_dataset), current_epoch
            )
        else:
            run.log("EM/train", current_epoch_EM * 100 / len(train_dataset))
        logger.info(
            f"Epoch {current_epoch} -> EM: {current_epoch_EM / len(train_dataset):.2%}"
        )

    if not args.azure:
        tb_writer.close()

    return global_step, train_loss / global_step
