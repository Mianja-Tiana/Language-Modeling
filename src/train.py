"""
Fine-tuning de GPT-2 sur un corpus texte prétraité (voir preprocess.py).
"""

from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments


def load_dataset(file_path, tokenizer, block_size=128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )


def load_data_collator(tokenizer, mlm=False):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm,
    )


def train(train_file_path, model_name, output_dir,
          overwrite_output_dir, per_device_train_batch_size,
          num_train_epochs, save_steps):

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    train_dataset = load_dataset(train_file_path, tokenizer)
    data_collator = load_data_collator(tokenizer)

    tokenizer.save_pretrained(output_dir)

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.save_pretrained(output_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    train(
        train_file_path="data/rakitra.txt",
        model_name="gpt2",
        output_dir="models/result",
        overwrite_output_dir=False,
        per_device_train_batch_size=2,
        num_train_epochs=5.0,
        save_steps=500,
    )
