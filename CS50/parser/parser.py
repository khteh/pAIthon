import nltk
import sys
import re
from nltk import Tree
"""
$ check50 --local ai50/projects/2024/x/parser
She never said a word <- N VP NP
until <- Conj
we were at the door here. <- N VP NP Adv

Holmes sat down <- N VP
and <- Conj
lit his pipe. <- V NP

i had a country walk on thursday and came home in a dreadful mess
i had a country walk on thursday -> N V NP P N
and <- Conj
came home in a dreadful mess -> V N P NP

https://realpython.com/python-nltk-sentiment-analysis/
"""
TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word" | "i"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""
NONTERMINALS = """
S -> N V | V N | N VP | NP N | NP VP | V NP | N V NP | V N P NP | S P N | S P NP | N VP NP | N VP NP Adv | S Conj S
NP -> Det N | Adj N | Det Adj N | N Conj N | P N | P Det N | P Adj N | P Det Adj N
VP -> V P | Adv V | V Adv
"""
nltk.download('punkt_tab')
grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)

def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    words = nltk.word_tokenize(sentence.lower())
    return [w for w in words if re.search('[a-zA-Z]', w)]

def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    result = []
    for node in tree.subtrees(filter=lambda t: t.label().endswith("NP")):
        subtree = list(node.subtrees(filter=lambda t: t.label().endswith("NP")))
        if len(subtree) == 1:
            result.append(node)
    return result

"""
Only executes when this file is run as a script
Does NOT execute when this file is imported as a module
__name__ stores the name of a module when it is loaded. It is set to either the string of "__main__" if it's in the top-level or the module's name if it is being imported.
"""
if __name__ == "__main__":
    main()
