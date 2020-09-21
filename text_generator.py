import pandas as pd
import markovify

corpus = pd.read_csv("./asset/eda.csv")["cleaned"].values
text_model = markovify.Text(corpus, state_size=3)
model_json = text_model.to_json()
# In theory, here you'd save the JSON to disk, and then read it back later.

reconstituted_model = markovify.Text.from_json(model_json)
reconstituted_model.make_short_sentence(280)
