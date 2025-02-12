import json
import math
import os
import bisect
import operator
import collections

# 3rd party
import numpy as np

CODE_MAP = str.maketrans("⬛🟨🟩", "012")

#################################Loading from files#################################
def load_word_weight(filename: str) -> np.ndarray:
    with open(filename, "r") as f:
        word_freqs = json.load(f)

    # sort by freq, low -> high 
    word_tuple = [(k, v) for k, v in word_freqs.items()]
    freq_sorted = sorted(word_tuple, key=operator.itemgetter(1))

    # generate score from the index, not by the actual frequency
    word_score = []
    for i, (word, _) in enumerate(freq_sorted):
        x = 0.002 * (i - 8000)
        word_score.append((word, 1 / (1 + math.exp(-x))))
    
    # go back to alphabetical order
    word_score = np.array([score for _, score in sorted(word_score)])

    return word_score

def load_words(filename: str) -> np.ndarray:
    words = []
    with open(filename, "r") as f:
        for line in f:
            words.append(line.strip())
    return np.array(words)

def load_matrix(word_list, filename="cache.npy") -> np.ndarray:
    # only compute if file doesn't exist
    if os.path.exists(filename):
        matrix = np.load(filename)
    else:
        print("Precomputing all possible combinations. This may take a few minutes")
        # 0-axis are the answers, 1-axis are the guesses
        matrix = np.zeros((len(word_list), len(word_list)), dtype=np.uint8)
        for i in range(len(word_list)):
            if i % 100 == 0:
                print(f"{100*i/len(word_list):.2f}")
            for j in range(len(word_list)):
                matrix[i,j] = get_pattern(word_list[i], word_list[j])

        np.save(filename, matrix)
    return matrix
#################################Loading from files#################################

##################################Util functions##################################
def binary_search(a, x) -> int:
    lo = 0
    hi = len(a)
    pos = bisect.bisect_left(a, x, lo, hi)                  
    return pos if pos != hi and a[pos] == x else -1


def get_pattern(answer:str, query_str:str) -> int:
    assert len(answer) == len(query_str)
    query = list(query_str)
    matched = list(answer)

    # green pass
    for i in range(len(query)):
        if query[i] == answer[i]:
            matched[i] = ''
            query[i] = '2'

    # yellow pass
    for i in range(len(query)):
        if query[i] in matched:
            matched[matched.index(query[i])] = ''
            query[i] = '1'

    # grey pass
    code = [num if num.isdigit() else '0' for num in query]

    # encode pattern as a base 3 number
    # 3^5 = 243, which conveniently fits into an 8bit number (0-255)
    code = "".join(code)
    code = int(code, 3)
    return code
##################################Util functions##################################

def get_entropies(possible_answers:list[int], matrix:np.ndarray, word_weights:np.ndarray) -> np.ndarray:
    entropies = np.zeros(matrix.shape[1], dtype=float)
    for j in range(matrix.shape[1]):
        # count unique patterns that result from each possible word
        counter = collections.defaultdict(float)
        for i in possible_answers:
            # words are weighted by their freq score
            counter[matrix[i, j]] += word_weights[i]

        counts = np.array(list(counter.values()))
        total = np.sum(counts)
        entropies[j] = np.sum(counts/total * np.log2(total/counts))

    return entropies

def run(guesses:list[str], patterns:list[str], word_weight:np.ndarray, word_list:np.ndarray, matrix:np.ndarray) -> None:
    # loop through inputs to get the current state of the game
    possible_answers = set(range(len(word_list)))
    for word, pattern in zip(guesses, patterns):
        word_index = binary_search(word_list, word)
        code = pattern.translate(CODE_MAP)

        indices = np.where(matrix[:, word_index] == int(code, base=3))[0]
        possible_answers &= set(indices.tolist())
    possible_answers = list(possible_answers)

    # calculate most likely answers
    percents = word_weight[possible_answers] / np.sum(word_weight[possible_answers])
    print(f"Answer Confidence (Total Entropy: {np.sum(percents * np.log2(1/percents)):.3f} bits):")
    best_candidates = sorted(zip(percents, word_list[possible_answers]), reverse=True)
    for percent, word, in best_candidates[:5]:
        print("\t", word, f"{100*percent:.2f}%")

    # calculate best words to cut down on word space
    entropies = get_entropies(possible_answers, matrix, word_weight)
    print(f"Expected Entropy Reduction:")
    best_clues = sorted(zip(entropies, word_list), reverse=True)
    for bits, word in best_clues[:5]:
        print("\t", word, f"{bits:.3f} bits")


if __name__ == "__main__":

    word_weight = load_word_weight("freq_map.json")
    word_list = load_words("allowed_words.txt")
    matrix = load_matrix(word_list)

    # Answer is 'lever'
    guesses = [
        "crane",
        "split",
        "exact",
        # "lover",
        ]
    patterns = [
        "⬛🟨⬛⬛🟨",
        "⬛⬛🟨⬛⬛",
        "🟨⬛⬛⬛⬛",
        # "🟩⬛🟩🟩🟩",
        ]

    run(guesses, patterns, word_weight, word_list, matrix)