import random
import pandas as pd
import numpy as np
import os
from transformers import TrOCRProcessor
from PIL import Image
import random
import imgaug.augmenters as iaa
import imageio
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision.models import resnet18
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer,AutoModelForTokenClassification,TokenClassificationPipeline,AutoFeatureExtractor
from transformers import VisionEncoderDecoderModel, AutoTokenizer
from transformers import TrOCRProcessor
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_metric
import evaluate
import warnings
from transformers import default_data_collator


warnings.filterwarnings(action='ignore')
CFG = {
    'IMG_HEIGHT_SIZE': 64,
    'IMG_WIDTH_SIZE': 224,
    'EPOCHS': 20,
    'LEARNING_RATE': 1e-3,
    'BATCH_SIZE': 256,
    'NUM_WORKERS': 0,  # 본인의 GPU, CPU 환경에 맞게 설정
    'SEED': 41
}
cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")
max_length = 64

def image_resize_1(image):
    aug1 = iaa.Resize({"height": 32, "width": image.shape[1]})
    return aug1(image=image)
def image_resize_2(image):
    aug1 = iaa.Resize({"height": image.shape[0], "width": 32})
    return aug1(image=image)

def my_Gnoise(image,su):
    #aug1 = iaa.Dropout(p=0.02) # 첫 번째 증강 기법 : Dropout #p = 0.05
    aug1 = iaa.AdditiveGaussianNoise(scale=(0, su*255),per_channel=False) # 두 번째 증강 기법 : GaussianBlur   #2~5
    #first_aug = aug2(image = image) # Dropout 적용
    return aug1(image = image) # GaussianBlur 적용 후 결과 반환
def my_Nnoise(image,su):
    o = image.shape[0]
    l = image.shape[1]
    if o<=31:
        image = image_resize_1(image=image)
    if l<=31:
        image = image_resize_2(image=image)
    aug1 = iaa.imgcorruptlike.DefocusBlur(severity=su) # 첫 번째 증강 기법 : Dropout #p = 0.05
    #aug1 = iaa.GaussianBlur(sigma=su)  # 두 번째 증강 기법 : GaussianBlur   #2~5
    # first_aug = aug2(image = image) # Dropout 적용

    return aug1(image=image)  # GaussianBlur 적용 후 결과 반환
def my_GNnoise(image,su,su2):
    o = image.shape[0]
    l = image.shape[1]
    if o <= 31:
        image = image_resize_1(image=image)
    if l <= 31:
        image = image_resize_2(image=image)
    aug1 = iaa.imgcorruptlike.DefocusBlur(severity=su2) # 첫 번째 증강 기법 : Dropout #p = 0.05
    aug2 = iaa.AdditiveGaussianNoise(scale=(0, su*255),per_channel=False)  # 두 번째 증강 기법 : GaussianBlur   #2~5
    first_aug = aug1(image = image) # Dropout 적용
    return aug2(image=first_aug)  # GaussianBlur 적용 후 결과 반환

def rotate(image,su):
    aug1 = iaa.Rotate((-1*su,su))
    return aug1(image=image)

class OCRDataset(Dataset):
    def __init__(self, root_dir, df, processor,mode='train',max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length
        self.mode = mode
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df['img_path'][idx]
        text = self.df['label'][idx]
        # prepare image (i.e. resize + normalize)
        if self.mode =='train':
            aug1 = random.random()
            aug2 = random.random()
            image = imageio.imread(self.root_dir + file_name)

            if aug1>0.7:
                su = random.randint(10, 20)
                image = rotate(image=image, su=su)

            if aug2>0.4:
                su = random.uniform(0.0, 0.06)
                noise_index = random.random()
                if noise_index > 0.3:
                    su2 = 2
                else:
                    su2 = 2
                image = my_GNnoise(image=image, su=su, su2=su2)

            image = Image.fromarray(image).convert("RGB")
        else:
            image = Image.open(self.root_dir + file_name).convert("RGB")


        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding


class OCRDataset_infer(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df['img_path'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text

        encoding = {"pixel_values": pixel_values.squeeze()}
        return encoding

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}

###########################





















if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    seed_everything(CFG['SEED'])  # Seed 고정

    df = pd.read_csv('./train.csv')
    df['len'] = df['label'].str.len()
    train_v1 = df[df['len'] == 1]
    df = df[df['len'] > 1]
    train_v2, val, _, _ = train_test_split(df, df['len'], test_size=0.1, random_state=CFG['SEED'])
    train = pd.concat([train_v1, train_v2])
    print(len(train), len(val))
    train_gt = [gt for gt in train['label']]
    train_gt = "".join(train_gt)
    letters = sorted(list(set(list(train_gt))))
    print(len(letters))
    vocabulary = ["-"] + letters
    print(len(vocabulary))
    idx2char = {k: v for k, v in enumerate(vocabulary, start=0)}
    char2idx = {v: k for k, v in idx2char.items()}
    df = train
    #train = pd.concat([train,val])
    print(len(train), len(val))

    train.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)

    #model = VisionEncoderDecoderModel.from_pretrained('daekeun-ml/ko-trocr-base-nsmc-news-chatbot')
    #tokenizer = AutoTokenizer.from_pretrained('daekeun-ml/ko-trocr-base-nsmc-news-chatbot')
    #processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

    #model = VisionEncoderDecoderModel.from_pretrained("C:/Users/tm011/Desktop/COMP/output_large/checkpoint-161028")
    #config = model.config
    #config.encoder.num_hidden_layers = 30
    #config.encoder.num_attention_heads = 18
    #config.encoder.hidden_dropout_prob = 0.1
    #config.decoder.dropout = 0.1
    #config.decoder.decoder_layers = 16
    #config.decoder.use_bfloat16 = True
    #config.encoder.use_bfloat16 = True
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
    #model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten",config=config)

    #processor = TrOCRProcessor.from_pretrained("C:/Users/tm011/Desktop/COMP/output_large/checkpoint-161028")

    train_dataset = OCRDataset(root_dir='',
                               df=train,
                               processor=processor,
                               mode='train'
                               )
    val_dataset = OCRDataset(root_dir='',
                             df=val,
                             processor=processor,
                             mode='val'
                             )

    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(val_dataset))

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = 2350

    # set beam search parameters
    # model.config.eos_token_id = processor.tokenizer.sep_token_id
    # model.config.max_length = 64
    model.config.early_stopping = True
    # model.config.no_repeat_ngram_size = 3
    # model.config.length_penalty = 2.0
    # model.config.num_beams = 4


    #model.config.decoder.encoder.add_cross_attention = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    training_args = Seq2SeqTrainingArguments(
        learning_rate=2e-6,
        predict_with_generate=True,
        evaluation_strategy="steps",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        bf16=True,
        output_dir="./first",
        logging_steps=100,
        save_steps=17892,
        eval_steps=17892,
        num_train_epochs=20,
        save_total_limit=50,
        dataloader_num_workers = 4
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=processor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    trainer.train()
