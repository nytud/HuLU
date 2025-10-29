import evaluate
import numpy as np
import torch
from torch import amp, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.modeling_outputs import (
    MultipleChoiceModelOutput,
    SequenceClassifierOutput,
)

from train.arguments import Arguments
from train.constants import CB_LABELS, CONJUNCTIONS, SST_LABELS
from train.helper import write_submission
from train.lora_helper import set_lora

accuracy_metric = evaluate.load("accuracy")
mcc_metric = evaluate.load("matthews_correlation")
f1_metric = evaluate.load("f1")


def compute_metrics(logits, labels):
    logits = logits.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    accuracy = accuracy_metric.compute(predictions=logits, references=labels)
    f1 = f1_metric.compute(predictions=logits, references=labels, average="weighted")
    mcc = mcc_metric.compute(predictions=logits, references=labels)

    return {
        "accuracy": accuracy["accuracy"],
        "mcc": mcc["matthews_correlation"],
        "f1": f1["f1"],
    }


class AutoForMultipleChoice(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

        self.classifier = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, num_choices, seq_length = input_ids.shape
        input_ids = input_ids.view(-1, seq_length)

        attention_mask = (
            attention_mask.view(-1, seq_length) if attention_mask is not None else None
        )
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        pooled_output = outputs.hidden_states[-1][:, -1, :]

        logits = self.classifier(pooled_output).view(batch_size, num_choices)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return MultipleChoiceModelOutput(loss=loss, logits=logits)
        return MultipleChoiceModelOutput(logits=logits)


class TrainPipeline:
    def __init__(self, hulu_args: Arguments, current_task: str, tokenizer_name: str):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.current_task = current_task
        self.hulu_args = hulu_args
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, clean_up_tokenization_spaces=True
        )
        self.train_loader, self.dev_loader, self.test_loader = None, None, None

    def collate_fn(self, batch):
        for item in batch:
            item["input_ids"] = torch.tensor(item["input_ids"], dtype=torch.long)
            item["attention_mask"] = torch.tensor(
                item["attention_mask"], dtype=torch.long
            )
            if "label" in item:
                item["label"] = torch.tensor(item["label"], dtype=torch.long)

        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        if not "label" in batch[0]:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        labels = (
            torch.stack([item["label"] for item in batch])
            if "label" in batch[0]
            else None
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": labels,
        }

    def set_tokenized_datasets(self, train_dataset, dev_dataset, test_dataset):
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.hulu_args.train_batch,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
        self.dev_loader = DataLoader(
            dev_dataset,
            batch_size=self.hulu_args.train_batch,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
        test_dataset = test_dataset.remove_columns("label")
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.hulu_args.train_batch,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def load_model(self):
        model_kwargs = (
            {"num_labels": 3}
            if self.current_task in ["sst", "cb"]
            else {"num_labels": 2}
        )
        if self.current_task == "copa":
            model = AutoForMultipleChoice(self.hulu_args.model_name)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.hulu_args.model_name, **model_kwargs
            )

        if self.hulu_args.use_lora:
            model = set_lora(self.hulu_args, sequente_classification=False, model=model)

        model.to(self.device)
        return model

    def training(self):
        model = self.load_model()

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

        num_eval_steps = len(self.train_loader) // 3
        step = 0

        for epoch in range(self.hulu_args.train_epochs):
            model.train()
            total_loss, correct_preds = 0, 0

            for batch in tqdm(
                self.train_loader,
                desc=f"Training Epoch {epoch + 1}/{self.hulu_args.train_epochs}",
            ):
                step += 1
                input_ids, attention_mask, labels = (
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                    batch["label"].to(self.device),
                )

                optimizer.zero_grad()

                with amp.autocast(device_type=device_type, enabled=use_fp16):
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )

                if use_fp16:
                    scaler.scale(output.loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    output.loss.backward()
                    optimizer.step()

                scheduler.step()
                total_loss += output.loss.item()
                correct_preds += (output.logits.argmax(dim=1) == labels).sum().item()

                if step % num_eval_steps == 0:
                    eval_loss, metrics = self.evaluate(model)
                    print(
                        f"Step {step}: Eval Loss = {eval_loss:.4f}, Eval Acc = {metrics['accuracy']}, Eval MCC = {metrics['mcc']}, Eval F1 = {metrics['f1']}"
                    )

            avg_loss = total_loss / len(self.train_loader)
            accuracy = correct_preds / len(self.train_loader.dataset)
            print(
                f"Epoch {epoch + 1}: Train Loss = {avg_loss:.4f}, Train Accuracy = {accuracy:.4f}"
            )

        return model

    def evaluate(self, model):
        model.eval()
        total_loss, correct_preds = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for _, batch in enumerate(self.dev_loader):
                input_ids, attention_mask, labels = (
                    batch["input_ids"].squeeze(1).to(self.device),
                    batch["attention_mask"].squeeze(1).to(self.device),
                    batch["label"].to(self.device),
                )

                output = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                total_loss += output.loss
                preds = output.logits.argmax(dim=1)
                correct_preds += (preds == labels).sum().item()

                all_preds.append(preds)
                all_labels.append(labels)

        avg_loss = total_loss / len(self.dev_loader)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = compute_metrics(all_preds, all_labels)
        return avg_loss, metrics

    def create_submission(self, model):
        model.eval()

        if self.current_task == "sst":
            reverse_labels = {v: k for k, v in SST_LABELS.items()}
        elif self.current_task == "cb":
            reverse_labels = {v: k for k, v in CB_LABELS.items()}
        else:
            reverse_labels = None

        predictions = []
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids, attention_mask = (
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                )
                output: MultipleChoiceModelOutput | SequenceClassifierOutput = model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                predictions.extend(output.logits.argmax(dim=-1).tolist())

        predictions_data = [
            {
                "id": str(i),
                "label": reverse_labels[pred] if reverse_labels else str(pred),
            }
            for i, pred in enumerate(predictions)
        ]

        write_submission(
            task=self.current_task,
            predictions_data=predictions_data,
            output_dir=self.hulu_args.output_dir,
        )
