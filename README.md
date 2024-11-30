# NeuralMachineTranslation-EnVi

Python script to translate English to Vietnamese general text file, with local AI model, use VinAI_Translate.

Aim to read translate general English articles with good accuracy and performance, offline.

### Description
Use VinAI_Translate model to translate En-Vi,
nltk PunktTokenizer to tokenize sentences in text file, and move into group or chunks of string with length about maxChar size,

Running uses GPU, quality and performance sounds good.

Sentences are move out of format and combined together. Please use '_chunks' and '_translated' files to compare before/after translation in details.

I have very low understanding with AI models... 
have not yet check carefully for missings, errors.
have not yet checked running for other environments.

Warning: !Please be careful with meaning changes from original contents made by translation tools.!

### Reference 

https://github.com/VinAIResearch/VinAI_Translate

https://github.com/dynamiccreator/voice-text-reader.git voice-text-reader

## Model

https://huggingface.co/vinai/vinai-translate-en2vi-v2

### Demo
COPS29-Trump.txt : After nearly 10 years of debate, COP29’s carbon trading deal is seriously flawed

https://theconversation.com/after-nearly-10-years-of-debate-cop29s-carbon-trading-deal-is-seriously-flawed-244493

### Install

(Not yet checked for details)

Install website, plus some modules:
```
pip install -r requirements.txt
```

Download model,

Run OK with python 3.12.

### Usage

```
python translate.py
```
```
python translate.py -f "COPS29-Trump.txt"
```

```
All options:
  -h, --help            show this help message and exit
  -f INPUT_PATH, --input_path INPUT_PATH
                        The path of the input text file to be read.
  -m MODEL, --model MODEL
                        The model path used for speak generation.
  -o OUT_PATH, --out_path OUT_PATH
                        The path of the translated text file.
```

### Code
Using the model directly:

GPU-based batch translation example
```
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer_en2vi = AutoTokenizer.from_pretrained("vinai/vinai-translate-en2vi-v2", src_lang="en_XX")
model_en2vi = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-en2vi-v2")
device_en2vi = torch.device("cuda")
model_en2vi.to(device_en2vi)

def translate_en2vi(en_texts: str) -> str:
    input_ids = tokenizer_en2vi(en_texts, padding=True, return_tensors="pt").to(device_en2vi)
    output_ids = model_en2vi.generate(
        **input_ids,
        decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["vi_VN"],
        num_return_sequences=1,
        num_beams=5,
        early_stopping=True
    )
    vi_texts = tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
    return vi_texts

# The input may consist of multiple text sequences, with the number of text sequences in the input ranging from 1 up to 8, 16, 32, or even higher, depending on the GPU memory.
en_texts = ["I haven't been to a public gym before.",
            "When I exercise in a private space, I feel more comfortable.",
            "i haven't been to a public gym before when i exercise in a private space i feel more comfortable"]
print(translate_en2vi(en_texts))
```

CPU-based sequence translation example
```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer_en2vi = AutoTokenizer.from_pretrained("vinai/vinai-translate-en2vi-v2", src_lang="en_XX")
model_en2vi = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-en2vi-v2")

def translate_en2vi(en_text: str) -> str:
    input_ids = tokenizer_en2vi(en_text, return_tensors="pt").input_ids
    output_ids = model_en2vi.generate(
        input_ids,
        decoder_start_token_id=tokenizer_en2vi.lang_code_to_id["vi_VN"],
        num_return_sequences=1,
        num_beams=5,
        early_stopping=True
    )
    vi_text = tokenizer_en2vi.batch_decode(output_ids, skip_special_tokens=True)
    vi_text = " ".join(vi_text)
    return vi_text

en_text = "I haven't been to a public gym before. When I exercise in a private space, I feel more comfortable."
print(translate_en2vi(en_text))
```

### License
