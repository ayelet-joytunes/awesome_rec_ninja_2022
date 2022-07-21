import implicit
from implicit.evaluation import train_test_split, mean_average_precision_at_k, precision_at_k
import pickle
from Data.data_fetch import fetch_data


class SvdModel:
    def __init__(self, train_percentage=0.1):
        self.profiles, self.songs, self.profile_plays = fetch_data()
        self.profile2index = {self.profiles[i]:i for i in range(len(self.profiles))}
        self.song2index = {self.songs[i]:i for i in range(len(self.songs))}
        self.profile_plays_train, self.profile_plays_test = train_test_split(self.profile_plays,train_percentage=train_percentage,random_state=42)



    def train(self):
        # initialize a model
        self.model = implicit.als.AlternatingLeastSquares(factors=50)

        # train the model on a sparse matrix of user/item/confidence weights
        self.model.fit(self.profile_plays_train)
        with open('model.sav', 'wb') as pickle_out:
            pickle.dump(self.model, pickle_out)
        return self

    def load_model(self):
        with open('model.sav', 'rb') as pickle_in:
            self.model = pickle.load(pickle_in)
        return self


    def recommend(self,profile_id):
        if self.model is None:
            raise ValueError("model should be loaded first")
        pofile_index = self.profile2index[profile_id]
        recommendations,scores = self.model.recommend(pofile_index, self.profile_plays[pofile_index])
        return [(self.songs[r],s) for r,s in zip(recommendations,scores)]

    def similar_songs(self,song_name):
        if self.model is None:
            raise ValueError("model should be loaded first")
        song_index = self.song2index[song_name]
        recommendations, scores = self.model.similar_items(song_index)
        return [(self.songs[r], s) for r, s in zip(recommendations, scores)]

    def evaluate(self):
        for k in range(1,11):
            print(f"precision at {k}: {precision_at_k(self.model,self.profile_plays_train, self.profile_plays_test, K=k)}")

# model = SvdModel().train()
# model = SvdModel().load_model()
# model.evaluate()
# print(model.similar_songs('Imagine_JohnLennon'))
# print(model.recommend("PFL61490B0ED843E5.88975979"))


