import os
from typing import Any, Dict

import torch
import torch.amp as amp
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from train.arguments import Arguments
from train.train import TrainPipeline, compute_metrics


def run_worker(
    rank: int,
    world_size: int,
    hulu_args: Arguments,
    current_task: str,
    tokenizer_name: str,
    dataset: Dict[str, Any],
) -> None:
    """
    Worker function for distributed training using FSDP.

    Sets up the environment and distributed process group,
    runs the training pipeline, creates the submission,
    and finally cleans up the process group.
    """
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    training_pipeline = FSdpPipeline(hulu_args, current_task, tokenizer_name)
    training_pipeline.set_tokenized_datasets(
        train_dataset=dataset["train"],
        dev_dataset=dataset["validation"],
        test_dataset=dataset["test"],
    )

    trained_model = training_pipeline.training()
    training_pipeline.create_submission(trained_model)

    dist.destroy_process_group()


class FSdpPipeline(TrainPipeline):
    """
    A training pipeline that extends TrainPipeline with Fully Sharded Data Parallel (FSDP) functionality.
    """

    def __init__(self, hulu_args: Arguments, current_task: str, tokenizer_name: str):
        super().__init__(hulu_args, current_task, tokenizer_name)

    def load_model(self):
        """
        Loads the base model and wraps it with FSDP.
        """
        model = super().load_model()
        return FSDP(model)

    def training(self):
        """
        Executes the training loop using FSDP. The loop supports mixed precision (fp16)
        training and periodically evaluates the model on the development set.
        """
        model = self.load_model().to(torch.cuda.current_device())

        use_fp16 = self.hulu_args.precision == "fp16"
        scaler = amp.GradScaler() if use_fp16 else None
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

        optimizer = AdamW(model.parameters(), lr=self.hulu_args.train_lr)
        total_steps = len(self.train_loader) * self.hulu_args.train_batch
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hulu_args.train_warmup,
            num_training_steps=total_steps,
        )
        gradient_accumulation_steps = self.hulu_args.gradient_accumulation_steps
        num_eval_steps = len(self.train_loader) // 3
        step = 0

        for epoch in range(self.hulu_args.train_epochs):
            model.train()
            total_loss, correct_preds = 0.0, 0

            for batch in tqdm(
                self.train_loader,
                desc=f"Training Epoch {epoch + 1}/{self.hulu_args.train_epochs}",
                disable=(dist.get_rank() != 0),
            ):
                input_ids = batch["input_ids"].to(torch.cuda.current_device())
                attention_mask = batch["attention_mask"].to(torch.cuda.current_device())
                labels = batch["label"].to(torch.cuda.current_device())

                with amp.autocast(device_type=device_type, enabled=use_fp16):
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

                if step % gradient_accumulation_steps == 0:
                    if use_fp16:
                        scaler.scale(output.loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        output.loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                scheduler.step()
                total_loss += output.loss.item()
                correct_preds += (output.logits.argmax(dim=1) == labels).sum().item()
                step += 1

                if step % num_eval_steps == 0 and dist.get_rank() == 0:
                    torch.cuda.empty_cache()
                    eval_loss, metrics = self.evaluate(model)
                    print(
                        f"Step {step}: Eval Loss = {eval_loss:.4f}, "
                        f"Eval Acc = {metrics['accuracy']}, "
                        f"Eval MCC = {metrics['mcc']}, "
                        f"Eval F1 = {metrics['f1']}"
                    )

            avg_loss = total_loss / len(self.train_loader)
            accuracy = correct_preds / len(self.train_loader.dataset)
            if dist.get_rank() == 0:
                print(
                    f"Epoch {epoch + 1}: Train Loss = {avg_loss:.4f}, "
                    f"Train Accuracy = {accuracy:.4f}"
                )

        return model

    def evaluate(self, model):
        """
        Evaluates the model on the development set and returns the average loss and computed metrics.
        """
        total_loss, correct_preds = 0.0, 0
        all_preds, all_labels = [], []
        model.eval()

        for batch in self.dev_loader:
            with torch.no_grad():
                input_ids = batch["input_ids"].to(torch.cuda.current_device())
                attention_mask = batch["attention_mask"].to(torch.cuda.current_device())
                labels = batch["label"].to(torch.cuda.current_device())
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                total_loss += outputs.loss.item()
                preds = outputs.logits.argmax(dim=1)
                correct_preds += (preds == labels).sum().item()

                all_preds.append(preds)
                all_labels.append(labels)

        avg_loss = total_loss / len(self.dev_loader)
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = compute_metrics(all_preds, all_labels)
        return avg_loss, metrics
