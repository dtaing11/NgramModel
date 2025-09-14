#!/usr/bin/env python3

import argparse
import pickle
import random
import re
import sys
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Iterable, Optional, Set


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

## Tokenize text, treating punctuation as separate tokens.
def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text)

 ## A n-gram model (bigram or trigram)
class NGramModel:
   
    def __init__(self, n: int):
        if n not in (2, 3):
            raise ValueError("n must be 2 (bigram) or 3 (trigram)")
        self.n: int = n
        self._ctxCounts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self._ctxProb: Dict[Tuple[str, ...], Tuple[List[str], List[float]]] = {}
        self.vocab: Set[str] = set()
        self._trained: bool = False
        self._backCounts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self._backProbs: Dict[Tuple[str, ...], Tuple[List[str], List[float]]] = {}

     ## Generate (context, next_token) pairs from token list.
    def contexts(self, tokens: List[str]) -> Iterable[Tuple[Tuple[str, ...], str]]:
       
        k = self.n - 1
        for i in range(len(tokens) - k):
            ctx = tuple(tokens[i:i + k])
            next_token = tokens[i + k]
            yield ctx, next_token

    ## Train the model on the given text data.
    def train(self, data: str) -> None:
        
        tokens = tokenize(data)
        self.vocab = set(tokens)
        if len(tokens) < self.n:
            sys.exit("Training data has fewer tokens than n; cannot train.")

        for ctx, next_token in self.contexts(tokens):
            self._ctxCounts[ctx][next_token] += 1

        self._ctxProb.clear()
        for ctx, counter in self._ctxCounts.items():
            total = sum(counter.values())
            toks = list(counter.keys())
            probs = [counter[t] / total for t in toks]
            self._ctxProb[ctx] = (toks, probs)
        
        # For trigram model, prepare bigram backoff probabilities
        if self.n == 3:
            self._backCounts.clear()
            k = 1  
            for i in range(len(tokens) - k):
                bctx = tuple(tokens[i:i + k])     
                bnxt = tokens[i + k]              
                self._backCounts[bctx][bnxt] += 1
            self._backProbs.clear()
            for bctx, bcounter in self._backCounts.items():
                btotal = sum(bcounter.values())
                btoks = list(bcounter.keys())
                bprobs = [bcounter[t] / btotal for t in btoks]
                self._backProbs[bctx] = (btoks, bprobs)

        self._trained = True



    ## Predict the next word given a context. If deterministic is True, return the most probable word.
    def predict_next_word(self, context: Tuple[str, ...], deterministic: bool = False) -> str:
        
       
        if not self._trained:
            raise ValueError("Model not trained/loaded.")

        expected_len = self.n - 1

        # Allow trigram to accept a 1-word context via bigram backoff
        if self.n == 3 and len(context) == 1:
            for w in context:
                if w not in self.vocab:
                    raise ValueError(f"Input token '{w}' not found in model vocabulary.")
            tokens, probs = self._backProbs[context]
            if deterministic:
                max_index = max(range(len(probs)), key=lambda i: probs[i])
                return tokens[max_index]
            return random.choices(tokens, weights=probs, k=1)[0]

        # Normal path: require exact n-1 words
        if len(context) != expected_len:
            raise ValueError(
                f"Expected {expected_len} prior word(s) for n={self.n}, got {len(context)}."
            )
        for w in context:
            if w not in self.vocab:
                raise ValueError(f"Input token '{w}' not found in model vocabulary.")
        tokens, probs = self._ctxProb[context]
        if deterministic:
            max_index = max(range(len(probs)), key=lambda i: probs[i])
            return tokens[max_index]
        return random.choices(tokens, weights=probs, k=1)[0]

    def generate(self, start: Tuple[str, ...], n_words: int, deterministic: bool = False) -> List[str]:
        ## Generate n_words starting from given start context
        k = self.n - 1
        for w in start:
            if w not in self.vocab:
                raise ValueError(f"Start token '{w}' not found in model vocabulary.")

        generated: List[str] = []
        ctx_list = list(start)

        for _ in range(n_words):
            ## Special case: trigram model with single-word start context
            if self.n == 3 and len(ctx_list) == 1:
                next_token = self.predict_next_word(tuple(ctx_list), deterministic=deterministic)
                generated.append(next_token)
                ctx_list = (ctx_list + [next_token])[-2:]
                continue

            next_token = self.predict_next_word(tuple(ctx_list), deterministic=deterministic)
            generated.append(next_token)
            if k > 0:
                ctx_list = (ctx_list + [next_token])[1:]

        return generated

    ## Read entire text file content.
def read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: could not find training data file '{path}'.")
        sys.exit(1)
       
## Save the model to a file using pickle.
def save_model(model: NGramModel, path: str) -> None:
    try:
        with open(path, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError as e:
        print(f"Error saving model to '{path}': {e}")
        

## Load the model from a file using pickle.
def load_model(path: str) -> NGramModel:
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, NGramModel):
            print("Error: Loaded object is not an NGramModel.")
            sys.exit(1)
        return obj
    except FileNotFoundError:
        print(f"Error: could not find model file at '{path}'.")
        sys.exit(1)
    except (pickle.UnpicklingError, OSError) as e:
        print(f"Error loading model from '{path}': {e}")
        sys.exit(1)


## Handle training activity.
def do_train(args: argparse.Namespace) -> None:  
    if args.n not in (2, 3):
        print("Error: --n must be 2 (bigram) or 3 (trigram).")
        sys.exit(1)
    if not args.data:
        print("Error: --data is required for 'train_ngram'.")
        sys.exit(1)
    if not args.save:
        print("Error: --save is required for 'train_ngram'.")
        sys.exit(1)

    text = read_text_file(args.data)
    model = NGramModel(n=args.n)
    try:
        model.train(text)
    except ValueError as e:
        print(f"Training error: {e}")
        sys.exit(1)

    save_model(model, args.save)
    print(f"Trained an {args.n}-gram model on '{args.data}' and saved to '{args.save}'.")


 ## Convert --word string into tuple of (n-1) words
def parse_start_words(arg_word: str, n: int) -> Tuple[str, ...]:
    required = n - 1
    parts = arg_word.strip().split()
    return tuple(parts)

## Handle prediction activity.
def do_predict(args: argparse.Namespace) -> None:
    
    model = load_model(args.load)
    try:
        start_ctx = parse_start_words(args.word, model.n)
        out_tokens = model.generate(start_ctx, n_words=args.nwords, deterministic=args.d)
    except ValueError as e:
        print(f"Prediction error: {e}", file=sys.stderr)
        sys.exit(1)

    print(" ".join(out_tokens))

## Build the command-line argument parser.
def build_arg_parser() -> argparse.ArgumentParser: 
    p = argparse.ArgumentParser(
        description="Bigram/Trigram N-gram model trainer and predictor (standard library only)."
    )
    p.add_argument("activity", choices=["train_ngram", "predict_ngram"],)
    p.add_argument("--data", type=str, default=None,)
    p.add_argument("--save", type=str, default=None,)
    p.add_argument("--load", type=str, default=None,)
    p.add_argument("--word", type=str, default=None,)
    p.add_argument("--nwords", type=int, default=None,)
    p.add_argument("--d", action="store_true",)
    p.add_argument("--n", type=int, choices=[2, 3], default=2,)
    return p

 ## Main entry point for command-line execution.
def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.activity == "train_ngram":
        do_train(args)
    elif args.activity == "predict_ngram":
        do_predict(args)
    else:
        print(f"Error: Unknown activity '{args.activity}'.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
