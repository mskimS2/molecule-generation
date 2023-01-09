"""
https://github.com/XinhaoLi74/SmilesPE/blob/master/SmilesPE/pretokenizer.py
"""

def atomwise_tokenizer(smi, exclusive_tokens=None):
    """
    Tokenize a SMILES molecule at atom-level:
        (1) 'Br' and 'Cl' are two-character tokens
        (2) Symbols with bracket are considered as tokens
    exclusive_tokens: A list of specifical symbols with bracket you want to keep. e.g., ['[C@@H]', '[nH]'].
    Other symbols with bracket will be replaced by '[UNK]'. default is `None`.
    """
    import re
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]

    if exclusive_tokens:
        for i, tok in enumerate(tokens):
            if tok.startswith('['):
                if tok not in exclusive_tokens:
                    tokens[i] = '[UNK]'
    return tokens


class Moltokenize(object):
    def tokenizer(self, sentence):
        return [tok for tok in atomwise_tokenizer(sentence) if tok != " "]


if __name__ == '__main__':

    smile = 'CCOC(=O)c1cncn1C1CCCc2ccccc21'
    print(Moltokenize().tokenizer(smile))
    # result:
    # ['C', 'C', 'O', 'C', '(', '=', 'O', ')', 'c', '1', 'c', 'n', 'c', 'n', '1', 'C', '1', 'C', 'C', 'C', 'c', '2', 'c', 'c', 'c', 'c', 'c', '2', '1']
