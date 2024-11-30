import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import sys
import re
from pathlib import Path
from textwrap import TextWrapper
import argparse

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import PunktTokenizer

model_path_default="vinai/vinai-translate-en2vi-v2" # Local model path or model name
input_path_default="input.txt" # text input file which is need to convert
output_path_default="_translated.txt" # translated Vi text file
maxChar = 300   # max size of chunk

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description="Translate English to Vietnamese general text file")
# Define optional parameters with values
parser.add_argument('-f','--input_path', type=str, help="The path of the input text file to be read.")
parser.add_argument('-m','--model', type=str, help="The model path used for speak generation.")
parser.add_argument('-o','--out_path', type=str, help="The path of the translated text file.")

# Parse the arguments
args = parser.parse_args()

# Access the arguments, using default values if not provided
param_input_path = args.input_path if args.input_path is not None else input_path_default
param_model = args.model if args.model is not None else model_path_default
param_out_path = args.out_path if args.out_path is not None else output_path_default

def load_text_file(file_path)-> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as reader:
            ## May add pre_process steps below. Remove empty lines,
            ## Store to a filename contains '_processed'
            processed_file = "_processed.txt"   # Use a default filename
            path = Path(file_path)
            processed_file = path.with_stem(''.join(path.stem) + "_processed")
            with open(processed_file, 'w', encoding='utf-8') as writer:
                for line in reader:
                    if line.strip():
                        writer.write(line)
            with open(processed_file, 'r', encoding='utf-8') as f:
                text = f.read()
                return text
    except FileNotFoundError:
        print("The file does not exist.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def capture_sentences(text, maxChar = 250, file_path = ""):
    """
        Tokenize sentence from text, remove trailing chars,
        split 1 sentence into 2 if len > maxChar,
        make chunks of combined sentences with len about <= maxChar
    """
    current_chunk = ""
    chunks = [] #list of combined sentences with total length under limit maxChar
    if not text:
        return np.array(chunks)

    # Tokenize sentence from string text file
    sentences = []
    sentences_tk = PunktTokenizer().tokenize(text.strip())  #nltk.sent_tokenize(text.strip())
#    print(f"--sentences_tk: {sentences_tk}")

#    wrapper = TextWrapper(width=maxChar/2)
    for idx in range (0, len(sentences_tk)):
        sentences_tk[idx] = sentences_tk[idx].replace("\n", " ")

        ## Not wrap a long sentences token to two
#        if len(sentences_tk[idx]) > maxChar:
#            wraps = wrapper.wrap(sentences_tk[idx])
#            sentences.extend([wrap for wrap in wraps])
#        else:
#            sentences.append(sentences_tk[idx])

        sentences.append(sentences_tk[idx])
#    print(f"--sentences: {sentences}")

    for idx in range (0, len(sentences)):
        sentences[idx] = sentences[idx].replace("\n", " ")
#        print(f"--sentences[{idx}]-len: {len(sentences[idx])} - {sentences[idx]}")

        ## Combine sentence, need to check
        ## chunks.append(sentences[idx])    #without combination
        if len(current_chunk) + len(sentences[idx]) > maxChar:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""
        current_chunk += " " + sentences[idx]
#        print(f"--current_chunk-len-idx,{idx}: {len(current_chunk)} - {current_chunk}")
    if current_chunk:
        chunks.append(current_chunk)    #Add the last current_chunk

#    print(f"\n--chunks: {chunks}")

    ## Save chunks to a filename contains '_chunks'
    processedFile = "_chunks.txt" # Use a default filename
    if file_path and Path(file_path):
        path = Path(file_path)
        processedFile = path.with_stem(''.join(path.stem) + "_chunks")
    with open(processedFile, 'w', encoding='utf-8') as writer:
        for idx in range (0, len(chunks)):
                writer.write(str(idx) + "\n")
                writer.write(chunks[idx] + "\n")
    print(f"--Saved sentence chunk file: {processedFile}---")

    return chunks

def initModel(model_path = model_path_default):
    tokenizer = AutoTokenizer.from_pretrained(model_path, src_lang="en_XX")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"--Done model loading, use device: {device}--")
    return model, tokenizer, device

def translate(input_texts: str, model, tokenizer, device: str) -> str:
    # The input may consist of multiple text sequences, with the number of text sequences in the input ranging from 1 up to 8, 16, 32, or even higher, depending on the GPU memory.
    #input_texts = ["I haven't been to a public gym before.",
    #            "When I exercise in a private space, I feel more comfortable.",
    #            "i haven't been to a public gym before when i exercise in a private space i feel more comfortable"]

#    input_texts = """
#    The long-feared Middle East
#    war is here. This is how
#    Israel could now hit back
#    at Iran
#    Ran Porat Published: October 3, 2024 3.29am BST
#    When Iran fired more than 180 ballistic missiles at Israel
#    this week in retaliation for the Israeli assassinations of the
#    Hamas and Hezbollah leaders, some were surprised by
#    Tehran’s forceful response. Sentence 1.
#    """

    input_ids = tokenizer(input_texts, padding=True, return_tensors="pt").to(device)
    output_ids = model.generate(
        **input_ids,
        decoder_start_token_id=tokenizer.lang_code_to_id["vi_VN"],
        num_return_sequences=1,
        num_beams=5,
        early_stopping=True
    )
    output_texts = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    output_texts = " ".join(output_texts)
    return output_texts

def mulTranslateAndSaveOutput(model, tokenizer, device, sentences, inputFile="", outputFile=output_path_default):
    """
    Save translated list of sentences to outputFile, with name include '_translated'
    """
    processedFile = output_path_default # Use a default filename
    if inputFile and Path(inputFile):
        path = Path(inputFile)
        processedFile = path.with_stem(''.join(path.stem) + "_translated")        
    with open(processedFile, 'w', encoding='utf-8') as writer:
        for idx in range (0, len(sentences)):
                print(f"--Translating {idx}/{len(sentences)}:{sentences[idx]}---")
                output_texts = translate(sentences[idx], model, tokenizer, device)
                writer.write(str(idx) + "\n")
                writer.write(output_texts + "\n")
                print(f"--Translated:{output_texts}---\n")

                ## without idx indication
                #writer.write(str(output_texts))
    print(f"--Done and Saved translated file:{processedFile}---")
    
if __name__ == "__main__":
    text_file = param_input_path
    text = load_text_file(text_file)
    sentences = capture_sentences(text, maxChar, text_file)
    print(f"--Got sentences list with size: {len(sentences)}")

    model, tokenizer, device = initModel()
    output_texts = mulTranslateAndSaveOutput(model, tokenizer, device, sentences, text_file, param_out_path)
