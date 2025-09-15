#!/usr/bin/env python3
import argparse, pickle, re, sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

 ## Read entire text file content.
def read_text_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: could not find training data file '{path}'.")
        sys.exit(1)
## Split text into whitespace-separated words.
def whitespace_words(text: str) -> List[str]:
    return [w for w in re.split(r"\s+", text.strip()) if w]

## Count pair frequency in the corpus 
def _get_stats(corpus: Dict[Tuple[str, ...], int]) -> Counter:
    pairs = Counter()
    for symbols, freq in corpus.items():
        if len(symbols) < 2:
            continue
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


# Single merge operation to the entire corpus.
def _merge_corpus(corpus: Dict[Tuple[str, ...], int],
                  pair: Tuple[str, str]) -> Dict[Tuple[str, ...], int]:
    a, b = pair
    merged: Dict[Tuple[str, ...], int] = {}
    for symbols, freq in corpus.items():
        out: List[str] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(a + b); i += 2
            else:
                out.append(symbols[i]); i += 1
        t = tuple(out)
        merged[t] = merged.get(t, 0) + freq
    return merged


##Byte Pair Encoding Model 
class BPEModel:
    def __init__(self) -> None:
        self.vocabulary: Set[str] = set()
        self.merges: List[Tuple[str, str]] = []
        self.token2id: Dict[str, int] = {}
        self.id2token: List[str] = []

## Train the BPE model with K merge 
    def train(self, text: str, k: int = 500) -> None:
        words = whitespace_words(text)
        corpus: Dict[Tuple[str, ...], int] = defaultdict(int)
        for w in words:
            corpus[tuple(w)] += 1

        self.merges.clear()
        for _ in range(k):
            stats = _get_stats(corpus)
            if not stats:
                break
            (a, b), freq = stats.most_common(1)[0]
            if freq < 1:
                break
            self.merges.append((a, b))
            corpus = _merge_corpus(corpus, (a, b))

    ## Count vocabulary items from the corpus
        vocab_counter = Counter()
        for symbols, freq in corpus.items():
            for s in symbols:
                vocab_counter[s] += freq
        self.vocabulary = set(vocab_counter.keys())

    ## Assign IDs based on the lenght of the token from longest to shortes, then Lexicographic
        sorted_vocab = sorted(self.vocabulary, key=lambda t: (-len(t), t))
        self.id2token = list(sorted_vocab)
        self.token2id = {tok: i for i, tok in enumerate(self.id2token)}


    ## Apply BPE merges to a word
    def _apply_to_word(self, word: str) -> List[str]:
        symbols = list(word)
        if not self.merges or len(symbols) < 2:
            return symbols
        rank = {pair: i for i, pair in enumerate(self.merges)}

        while True:
            if len(symbols) < 2:
                break
            best = None
            for i in range(len(symbols) - 1):
                p = (symbols[i], symbols[i + 1])
                if p in rank and (best is None or rank[p] < rank[best[1]]):
                    best = (i, p)
            if best is None:
                break
            a, b = best[1]
            i = 0; merged: List[str] = []
            while i < len(symbols):
                if i < len(symbols)-1 and symbols[i]==a and symbols[i+1]==b:
                    merged.append(a + b); i += 2
                else:
                    merged.append(symbols[i]); i += 1
            symbols = merged
        return symbols

## Tokenize the input text using trained BPE
    def tokenize(self, text: str) -> Tuple[List[str], List[int]]:
        tokens_out: List[str] = []
        for w in whitespace_words(text):
            word_tokens = self._apply_to_word(w)
            for t in word_tokens:
                if t in self.token2id:
                    tokens_out.append(t)
                else:
                    for ch in t:
                        tokens_out.append(ch)
        ids = [self.token2id[t] for t in tokens_out]
        return tokens_out, ids

    ## Seralize Data model to byte for saving
    def to_bytes(self) -> bytes:
        return pickle.dumps({
            "vocabulary": list(self.vocabulary),
            "merges": self.merges,
            "token2id": self.token2id,
            "id2token": self.id2token,
        }, protocol=pickle.HIGHEST_PROTOCOL)
    

    ## Seralize BPE Model from bytes
    @classmethod
    def from_bytes(cls, data: bytes) -> "BPEModel":
        payload = pickle.loads(data)
        obj = cls()
        obj.vocabulary = set(payload["vocabulary"])
        obj.merges = list(payload["merges"])
        obj.token2id = dict(payload["token2id"])
        obj.id2token = list(payload["id2token"])
        return obj
## Save the model to a file using pickle.    
def save_model(model: BPEModel, path: str) -> None:
    with open(path, "wb") as f:
        f.write(model.to_bytes())
## Load the model from a file using pickle.
def load_model(path: str) -> BPEModel:
    with open(path, "rb") as f:
        return BPEModel.from_bytes(f.read())

## start the training and save
def do_train(args: argparse.Namespace) -> None:
    text = read_text_file(args.data)
    model = BPEModel()
    model.train(text, k=args.k)
    save_model(model, args.save)
    print(f"Trained BPE with {len(model.vocabulary)} tokens and {len(model.merges)} merges. Saved to '{args.save}'.")

## tokenize input text using trained BPE 
def do_tokenize(args: argparse.Namespace) -> None:
    model = load_model(args.load)
    toks, ids = model.tokenize(args.text)
    print(toks)
    print(ids)


## Build the command-line argument parser.
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BPE (no EOW) trainer and tokenizer")
    p.add_argument("activity", choices=["train_bpe", "tokenize"])
    p.add_argument("--data", type=str, help="Path to training text (train_bpe)")
    p.add_argument("--save", type=str, help="Where to save model (train_bpe)")
    p.add_argument("--load", type=str, help="Where to load model (tokenize)")
    p.add_argument("--text", type=str, help="Text to tokenize (tokenize)")
    p.add_argument("--k", type=int, default=500, help="Merge iterations (default 500)")
    return p

 ## Main entry point for command-line execution.
def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    if args.activity == "train_bpe":
        do_train(args)
    else:
        do_tokenize(args)

if __name__ == "__main__":
    main()
