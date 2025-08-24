import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import json

class FTSystem:
    def __init__(self, model_name="distilgpt2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def prepare_dataset(self, qa_data_path):
        """
        Prepares the Q&A dataset for supervised instruction fine-tuning.
        """
        with open(qa_data_path, 'r') as f:
            qa_pairs = json.load(f)
        
        dataset_dicts = []
        for pair in qa_pairs:
            instruction_text = f"### Instruction: Based on the provided financial data, answer the following question. ### Question: {pair['question']} ### Answer: {pair['answer']}"
            dataset_dicts.append({"text": instruction_text})
        
        return Dataset.from_dict({"text": [d["text"] for d in dataset_dicts]})

    def fine_tune(self, dataset, output_dir="./models/fine_tuned_model"):
        """
        Fine-tunes the model on the prepared dataset.
        """
        tokenized_dataset = dataset.map(
            lambda examples: self.tokenizer(examples['text'], truncation=True, padding="max_length"),
            batched=True
        )

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        )
        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print("Fine-tuning complete. Model saved.")
    
    def get_ft_answer(self, query):
        prompt = f"### Instruction: Based on the provided financial data, answer the following question. ### Question: {query} ### Answer:"
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=512,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("Answer:")[1].strip()
        confidence = 0.95 # Placeholder confidence, could be based on probability distribution
        return answer, confidence