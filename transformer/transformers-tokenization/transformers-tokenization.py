class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0

        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        self.word_to_id = {
            self.pad_token: 0,
            self.unk_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }
    
        words = set()
    
        for text in texts:
            for word in text.lower().split():
                words.add(word)
    
        for word in sorted(words):
            self.word_to_id[word] = len(self.word_to_id)
    
        self.id_to_word = {
            idx: word
            for word, idx in self.word_to_id.items()
        }
    
        self.vocab_size = len(self.word_to_id)

    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        return [
            self.word_to_id.get(word, self.word_to_id[self.unk_token])
            for word in text.lower().split()
        ]

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        return " ".join(
            self.id_to_word.get(i, self.unk_token)
            for i in ids
        )