import whisper
from whisper.tokenizer import get_tokenizer
import torch
from scipy.io.wavfile import read, write
import numpy as np
import pandas as pd
import math
import glob
import random
import copy
import os
from scipy.stats import entropy
import shutil
from tqdm import tqdm
import re
import argparse

def read_audio_to_buffer(filepath, slice_len):
    audio = read(filepath)[1]        
    if audio.shape[0] < slice_len:
        audio = np.pad(audio, (0, slice_len - audio.shape[0]))
    audio = audio[:slice_len]

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32767
    elif audio.dtype == np.float32:
        pass
    else:
        raise NotImplementedError('Scipy cannot process atypical WAV files.')
    #audio /= np.max(np.abs(audio))
    return(audio)

def eval_labeled_file(path, model, tokenizer, candidates, prompt):
    df = get_censored_probs(model, tokenizer, path, candidates, prompt)
    recovered = df.iloc[0].phrase 
    recovered_prob = df.iloc[0].prob
    entropy_val = entropy(df.prob)
    
    return {
        'recovered' : recovered,
        'recovered_prob' : recovered_prob,
        'entropy' : entropy_val,
        'path': path
    }
    
    # return({'correct':correct, 'label':label, 
    #        'recovered': recovered, 'rank_of_correct': rank_of_correct,
    #        'surp_of_correct':surp_of_correct, 'prob_of_correct':prob_of_correct, 'entropy': entropy_val, 'path': path})

    # return({'correct':correct, 'label':label, 'label1':label1, 'label2':label2, 
    #        'recovered': recovered, 'recovered1': recovered1, 'recovered2': recovered2, 'rank_of_correct': rank_of_correct,
    #        'surp_of_correct':surp_of_correct, 'prob_of_correct':prob_of_correct, 'entropy': entropy_val, 'path': path})

    
def get_censored_probs(model, tokenizer, wav_path, candidates, prompt=None):
    audio = read_audio_to_buffer(wav_path, 20480)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to('cuda')

    # Add a batch dimension to the mel spectrogram
    mel = mel.unsqueeze(0)  # Shape: (1, n_mels, n_frames)

    # Prepare your set of phrases
    log_likelihoods = []

    for phrase in candidates:
        # Encode the phrase into tokens
    
        # this is a whisper model
        # Add start and end tokens if necessary
        tokens = tokenizer.encode(phrase)
        tokens = [tokenizer.sot_sequence[0]] + tokens + [tokenizer.eot]            
        
        if prompt is not None:
            tokens = [tokenizer.sot_prev] + tokenizer.encode(prompt) + tokens            

        tokens = torch.tensor([tokens]).to('cuda')  # Shape: (1, seq_len)


        with torch.no_grad():
            # Forward pass through the model
            outputs = model(mel, tokens)        

            logits = outputs  # Shape: (1, seq_len, vocab_size)
            
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            # Compute the log-likelihood of the phrase
            # Shift tokens to align with the outputs
            target_tokens = tokens[:, 1:]  # Exclude the first token (sot_sequence)
            log_probs = log_probs[:, :-1, :]  # Exclude the last output to match target_tokens
            
            # Gather the log probabilities of the target tokens
            token_log_probs = torch.gather(log_probs, 2, target_tokens.unsqueeze(-1)).squeeze(-1)

            # Sum the log probabilities
            log_likelihood = token_log_probs.sum().item()
            log_likelihoods.append(log_likelihood)

    # Handle numerical stability
    max_log_likelihood = max(log_likelihoods)
    adjusted_log_likelihoods = [ll - max_log_likelihood for ll in log_likelihoods]

    # Exponentiate
    unnormalized_probs = [math.exp(ll) for ll in adjusted_log_likelihoods]

    # Normalize the probabilities
    total = sum(unnormalized_probs)
    probabilities = [up / total for up in unnormalized_probs]

    # Output the probability distributionc
    rdf = pd.DataFrame({'phrase':candidates,'prob':probabilities})
    rdf = rdf.sort_values('prob', ascending=False)
    
    return(rdf)




if __name__ == "__main__":

    parser = argparse.ArgumentParser()    

    parser.add_argument(
        '--whisper_model',
        type=str,
        default="medium.en",
        help="What model should I use? tiny.en and medium.en are known to work"
    )     

    parser.add_argument(
        '--input_dir',
        type=str,    
        help="path to directory with wavs"
    )     


    parser.add_argument(
        '--output_dir',
        type=str,    
        help="output directory to place results.csv"
    )     

    parser.add_argument(
        '--vocab',
        type=str,    
        help="semicolon-separated list of vocabulary items"
    )     

    parser.add_argument(
        '--wav_suffix',
        type=str,    
        default = '',
        help="suffix to select appropriate waves in the glob, e.g. 'converted'"
    )     

    args = parser.parse_args()


    vocab = args.vocab.split(';')
    full_list = random.sample(vocab, len(vocab)) + random.sample(vocab, len(vocab)) + random.sample(vocab, len(vocab))
    prompt = ', '.join(full_list)+', '
    print(prompt)
    candidates = vocab

    whisper_model = whisper.load_model(args.whisper_model)
    if args.whisper_model in ['tiny.en','medium.en']:
        tokenizer  = get_tokenizer(multilingual=False)

    else:
        raise NotImplementedError('Unclear what tokenizer to use outside of these models')


    print('Processing '+args.input_dir+'...')

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    files_to_process  = np.array(glob.glob(os.path.join(args.input_dir,'*'+args.wav_suffix+'.wav'))) 

    results = []
    for i in tqdm(range(len(files_to_process))):
        file_to_process = files_to_process[i]
        results.append(eval_labeled_file(file_to_process, whisper_model, tokenizer, candidates, prompt))

    results_df = pd.DataFrame(results)
    output_path = os.path.join(args.output_dir,'whisper_transcription_results.csv')
    results_df.to_csv(output_path)
    print('Complete! Inspect results at '+output_path)