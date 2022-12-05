# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import gensim.downloader as dl
import numpy as np


def count_same_occurence(tw_top, wiki_top):
    count = 0
    tw_top = [i[0] for i in tw_top]
    wiki_top = [i[0] for i in wiki_top]
    for word in tw_top:
        if word in wiki_top:
            count += 1
    return count


def print_hi(name):
    model = dl.load("word2vec-google-news-300")
    vocab = model.index_to_key
    twitter_model = dl.load("glove-twitter-200")
    wiki_model = dl.load("glove-wiki-gigaword-200")

    ## 15 similar word in twitter and wiki
    similar_words = ["blue", "red", "dog", "soccer", "israel", "green", "orange", "pepper", "yellow"]
    diff_words = ["bamba","number","trump", "musk", "caption", "coke", "koala", "bug", "hair", "matrix", "furry", "corny", "water", "binary"]

    sim_list_word = [ "factor", "far", "fine", "flood", "fox", "function", "general", "gentle", "grace", "grand", "grip", "gross", "grow", "gum", "hammer", "hole", "idle", "inactive", "indifferent", "inside", "interest", "introduce", "james", "king", "knot", "labor", "lean", "left", "lie", "little", "living", "lodge", "lose", "lost", "love", "maintain", "make out", "mantle", "moderate", "nail", "narrow", "pace", "paddle", "panel", "passage", "pattern", "peg", "perch", "plant", "plunge", "poke", "pot", "practice", "program", "projection", "purge", "put", "out", "race", "radiate", "rally", "rap", "ray", "regard", "regenerate", "representation", "romance", "runner", "scene", "school", "scrape", "screw", "share", "sheet", "single", "sit", "skim", "slice", "slide", "slug", "smell", "smith", "smoke", "softness", "space", "spat", "special", "speed", "spell", "spill", "spiral", "spur", "stage", "stake", "steady", "stem", "sting", "stream", "strong", "submit", "suit", "sure", "surface", "tag", "tongue", "translate", "tread", "treat", "try", "under", "undercut", "vote", "water", "waver", "weight", "west", "wish", "yoke", "zero"]
    # matrix
    same = []
    dif = []
    for word in vocab:
        tw_most, wik_most = twitter_model.most_similar(word), wiki_model.most_similar(word)
        temp = count_same_occurence(tw_most, wik_most)
        if temp >= 9:
            same.append(word)
        if temp <= 1:
            dif.append(word)
    print("---------------------similar---------------")
    print(same)
    print("---------------------different-------------")
    print(dif)



    #
    # for word in similar_words:
    #     print(word)
    #     tw_most, wik_most = twitter_model.most_similar(word), wiki_model.most_similar(word)
    #     print(tw_most)
    #     print(wik_most)
    #     print(count_same_occurence(tw_most, wik_most))
    #     print("\n\n")
    #
    # print("-------------------------different-------------------------------")
    # for word in diff_words:
    #     print(word)
    #     tw_most, wik_most = twitter_model.most_similar(word), wiki_model.most_similar(word)
    #     print(tw_most)
    #     print(wik_most)
    #     print(count_same_occurence(tw_most, wik_most))
    #     print("\n\n")

    # #workikng
    # cosine_similarity_w1_w2 = numpy.dot(model["small"], model["humongous"]) / (
    #             numpy.linalg.norm(model["small"]) * numpy.linalg.norm(model["humongous"]))
    #
    # cosine_similarity_w1_w3 = numpy.dot(model["small"], model["large"]) / (
    #             numpy.linalg.norm(model["small"]) * numpy.linalg.norm(model["large"]))
    #
    # # w1 - small | w2 = humongous | w3 = large
    # print(str(cosine_similarity_w1_w2) + " < " + str(cosine_similarity_w1_w3))

    # #workikng
    # cosine_similarity_w1_w2 = numpy.dot(model["open"], model["agape"]) / (
    #             numpy.linalg.norm(model["open"]) * numpy.linalg.norm(model["agape"]))
    #
    # cosine_similarity_w1_w3 = numpy.dot(model["open"], model["close"]) / (
    #             numpy.linalg.norm(model["open"]) * numpy.linalg.norm(model["close"]))
    #
    # # w1 - open | w2 = agape | w3 = close
    # print(str(cosine_similarity_w1_w2) + " < " + str(cosine_similarity_w1_w3))
    #
    #
    # #workikng
    # cosine_similarity_w1_w2 = numpy.dot(model["same"], model["equivalent"]) / (
    #             numpy.linalg.norm(model["same"]) * numpy.linalg.norm(model["equivalent"]))
    #
    # cosine_similarity_w1_w3 = numpy.dot(model["same"], model["different"]) / (
    #             numpy.linalg.norm(model["same"]) * numpy.linalg.norm(model["different"]))
    #
    # # w1 - same | w2 = equivalent | w3 = different
    # print(str(cosine_similarity_w1_w2) + " < " + str(cosine_similarity_w1_w3))

    # # PART 1
    # #1
    # print(model.most_similar("fish"))
    #
    #
    # #1
    # print(model.most_similar("train"))
    #
    # #1
    # print(model.most_similar("grave"))
    #
    # #2
    # print(model.most_similar("can"))
    #
    # #2
    # print(model.most_similar("blue"))
    #
    # #2
    # print(model.most_similar("punch"))


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    print_hi("PyCharm")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
