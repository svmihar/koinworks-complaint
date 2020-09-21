from dataclasses import dataclass
import pandas as pd
import json
import markovify
try:
    from fire import Fire
except:
    pass


def train():
    corpus = pd.read_csv("./asset/eda.csv")
    corpus.dropna(inplace=True)
    corpus['wc'] = corpus['cleaned'].apply(lambda x: len(x.split()))
    corpus = corpus[corpus['wc']>2]
    corpus = corpus[corpus['complaint']==1]["cleaned"].values
    text_model = markovify.Text(corpus, state_size=2)
    model_json = text_model.to_json()
    with open("./asset/markov.json", "w") as f:
        json.dump(model_json,f )


@dataclass
class generator():
    model_json: str='./asset/markov.json'
    def __post_init__(self):
        json_object = json.load(open(self.model_json))
        self.model = markovify.Text.from_json(json_object)
        self.model.compile()

    def generate(self):
        return self.model.make_short_sentence(280)

if __name__ == "__main__":
    train()
    test_gen =  generator()
    print(test_gen.generate())