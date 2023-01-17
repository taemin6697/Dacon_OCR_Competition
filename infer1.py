import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from main import OCRDataset_infer, max_length
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer
from transformers import AutoTokenizer,AutoModelForTokenClassification,TokenClassificationPipeline,AutoFeatureExtractor


if __name__ == '__main__':
    #model = VisionEncoderDecoderModel.from_pretrained('C:/Users/tm011/Desktop/COMP/trocr_large_add_data_total_GNnoise/checkpoint-83676')
    #tokenizer = AutoTokenizer.from_pretrained('C:/Users/tm011/Desktop/COMP/trocr_large_add_data_total_GNnoise/checkpoint-83676')

    model = VisionEncoderDecoderModel.from_pretrained(
        './first/checkpoint-250488')
    tokenizer = AutoTokenizer.from_pretrained(
        './first/checkpoint-250488')
    #processor = TrOCRProcessor.from_pretrained('C:/Users/tm011/Desktop/COMP/output_beit_ko/checkpoint-10')
    #feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/beit-base-patch16-224")
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten',size=384)

    # vision_hf_model = 'timm/beitv2_base_patch16_224.in1k_ft_in22k_in1k'
    # nlp_hf_model = "klue/roberta-large"
    # model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(vision_hf_model, nlp_hf_model)
    # tokenizer = AutoTokenizer.from_pretrained(nlp_hf_model)
    # processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten', size=224)


    test_df= pd.read_csv('./test.csv')

    test_dataset = OCRDataset_infer(root_dir ='',
                                   df=test_df,
                                    processor=processor
                              )
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False,num_workers=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    pred = []
    with torch.no_grad():
        for batch_id, x in enumerate(tqdm(test_dataloader)):
            #pixel_values = (processor(image, return_tensors="pt").pixel_values).to(device)
            pixel_values = x['pixel_values'].to(device)
            generated_ids = model.generate(pixel_values)
            generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True,max_length=max_length)
            print(generated_text)

            pred.extend(generated_text)

    submit = pd.read_csv('./sample_submission.csv')
    submit['label'] = pred
    submit.to_csv('./first_check_point-250488.csv', index=False)
