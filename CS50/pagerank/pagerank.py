import os
import random, operator
import re
import sys
import math
from math import inf
from collections import defaultdict
DAMPING = 0.85
SAMPLES = 10000
"""
$ pipenv run check50 --local ai50/projects/2024/x/pagerank
"""
def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    print(f"corpus: {corpus}")
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.

    For example, if the corpus were {"1.html": {"2.html", "3.html"}, "2.html": {"3.html"}, "3.html": {"2.html"}}, the page was "1.html", and the damping_factor was 0.85, then the output of transition_model should be {"1.html": 0.05, "2.html": 0.475, "3.html": 0.475}. 
    This is because with probability 0.85, we choose randomly to go from page 1 to either page 2 or page 3 (so each of page 2 or page 3 has probability 0.425 to start), but every page gets an additional 0.05 because with probability 0.15 we choose randomly among all three of the pages.    
    """
    result = dict()
    if len(corpus[page]) == 0:
        result = {key: 1/len(corpus) for key in corpus}
    else:
        result = {key: (1 - damping_factor) / len(corpus) for key in corpus}
        url_probability = damping_factor / len(corpus[page])
        for p in corpus[page]:
            result[p] += url_probability
    return result

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.

    Our random surfer now starts by choosing a page at random, and then, for each additional sample we’d like to generate, chooses a link from the current page at random with probability d, and chooses any page at random with probability 1 - d. 
    If we keep track of how many times each page has shown up as a sample, we can treat the proportion of states that were on a given page as its PageRank.
    """
    page = random.choice(list(corpus.keys()))
    result = {key:0 for key in corpus}
    result[page] = 1
    for i in range(n):
        next = transition_model(corpus, page, damping_factor)
        random_next = random.choices(list(next.keys()), weights=list(next.values()), k=1)
        page = random_next[0]
        result[page] = result.get(page, 0) + 1
    total = sum(result.values())
    for k in result:
        result[k] /= total
    return result

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.

    A page that has no links at all should be interpreted as having one link for every page in the corpus (including itself).
    """
    result = {key: 1/len(corpus) for key in corpus}
    urls = dict()
    references = {key: set() for key in corpus}
    for k,v in corpus.items():
        if len(v) > 0:
            urls[k] = len(v)
        else:
            urls[k] = len(corpus)
        if len(v) > 0:
            for p in v:
                references[p].add(k)
        else:
            for p,v in references.items():
                v.add(k)
    print(f"corpus: {corpus}")
    print(f"references: {references}")
    print(f"urls: {urls}")
    diff = inf
    d = (1 - damping_factor) / len(corpus)
    while diff > 0.001:
        diff1 = set()
        for page,v in references.items():
            rank = d + damping_factor * sum(result[k] / urls[k] for k in v)
            diff1.add(abs(rank - result[page]))
            result[page] = rank
        diff = max(diff1)
    return result
"""
Only executes when this file is run as a script
Does NOT execute when this file is imported as a module
__name__ stores the name of a module when it is loaded. It is set to either the string of "__main__" if it's in the top-level or the module's name if it is being imported.
"""
if __name__ == "__main__":
    main()
