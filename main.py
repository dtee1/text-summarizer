from huggingface_hub import interpreter_login
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import __version__, BartTokenizer, BartForConditionalGeneration
import evaluate
import wandb
from torch import cuda
import time

# wandb.login()
interpreter_login()
device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

wandb.init(project="BART_summarization")

config = wandb.config
config.TRAIN_BATCH_SIZE = 2
config.VALID_BATCH_SIZE = 2
config.TRAIN_EPOCHS = 2
config.LEARNING_RATE = 1e-4
config.SEED = 42
config.MAX_LEN = 512
config.SUMMARY_LEN = 150 

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
torch.backends.cudnn.deterministic = True

new_repo = "test"
repo_name = "EducativeCS2023/bart-base-summarization"

df = pd.read_csv('BBCarticles.csv' ,encoding='latin-1')
df = df[['Text','Summary']]
df.Text = 'summarize: ' + df.Text

split = 0.025
train_dataset=df.sample(frac=split,random_state = config.SEED)
eval_dataset=df.drop(train_dataset.index).sample(frac=split,random_state = config.SEED).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

print("train_dataset", train_dataset.shape)
print("eval_dataset", eval_dataset.shape)

df.head(3)

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.Summary = self.data.Summary
        self.Text = self.data.Text

    def __len__(self):
        return len(self.Summary)

    def __getitem__(self, index):
        Text = str(self.Text[index])
        Text = ' '.join(Text.split())

        Summary = str(self.Summary[index])
        Summary = ' '.join(Summary.split())

        source = self.tokenizer([Text], max_length= self.source_len, padding='max_length', truncation=True, return_tensors='pt')
        target = self.tokenizer([Summary], max_length= self.summ_len, padding='max_length', truncation=True, return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long)
        }
    

tokenizer = BartTokenizer.from_pretrained(repo_name)

tokenizer.push_to_hub(new_repo)

training_set = CustomDataset(train_dataset, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)
eval_set = CustomDataset(eval_dataset, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)

training_loader = DataLoader(training_set, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)
eval_loader = DataLoader(eval_set, batch_size=config.VALID_BATCH_SIZE, shuffle=False, num_workers=0)

model = BartForConditionalGeneration.from_pretrained(repo_name)
model = model.to(device)

optimizer = torch.optim.Adam(params =  model.parameters(), lr=config.LEARNING_RATE)

wandb.watch(model, log="all")

def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    for _,data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype = torch.long)
        y_ids = y[:, :-1].contiguous()
        labels = y[:, 1:].clone().detach()
        labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype = torch.long)
        mask = data['source_mask'].to(device, dtype = torch.long)

        outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=labels)
        loss = outputs[0]
        if _%10 == 0:
            wandb.log({"Training Loss": loss.item()})

        if _%500==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
for epoch in range(config.TRAIN_EPOCHS):
    train(epoch, tokenizer, model, device, training_loader, optimizer)

model.push_to_hub(new_repo)

def predict(tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(
                input_ids = ids,
                attention_mask = mask, 
                max_length=150, 
                num_beams=2,
                repetition_penalty=2.5, 
                length_penalty=1.0, 
                early_stopping=True
                )
            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            if _%100==0:
                print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals

tik = time.time()
predictions, actuals = predict(tokenizer, model, device, eval_loader)
results = pd.DataFrame({'predictions':predictions,'actuals':actuals})
results.to_csv('predictions.csv')
tok = time.time()
print("time taken", tok-tik, "seconds")
results.head()
rouge_score = evaluate.load("rouge")

scores = rouge_score.compute(
    predictions=results['predictions'], references=results['actuals']
)
pd.DataFrame([scores]).T.head()