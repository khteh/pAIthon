import csv
import itertools
import sys
from functools import reduce
from operator import mul
"""
$ pipenv run check50 --local ai50/projects/2024/x/heredity
"""
PROBS = {
    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },
    # Conditional probability that a person exhibits a trait (like hearing impairment).
    "trait": {
        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },
        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },
        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },
    # Mutation probability - the probability that a gene mutates from being the gene in question to not being that gene, and vice versa.
    "mutation": 0.01
}


def main():
    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }
    print(f"probabilities: {probabilities}")
    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data

def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]
"""
When a parent has the trait, the probability of passing it on is not simply 0.5 * PROBS["mutation"]. It's a bit more complex. You need to consider two scenarios: the parent passes on the gene (50% chance) and it doesn't mutate (99% chance), or the parent doesn't pass on the gene (50% chance) and it does mutate (1% chance).
If a parent doesn't have the trait, you also need to consider two scenarios: the parent passes on the gene (which can't happen normally, so 0% chance) and it doesn't mutate (99% chance), or the parent doesn't pass on the gene (100% chance) and it does mutate (1% chance).
"""
def probability(people, one_gene, two_genes, have_trait, person):
    gene = (2 if person in two_genes else 1 if person in one_gene else 0)
    if (not people[person]["mother"] or people[person]["mother"] not in people) and (not people[person]["father"] or people[person]["father"] not in people):
        return PROBS["gene"][gene] * PROBS["trait"][gene][person in have_trait]
    else:
        mother = people[person]["mother"]
        father = people[person]["father"]
        m_gene = 1
        if mother in one_gene: # 50% inheritance
            m_gene = 0.5
        elif mother in two_genes:
            m_gene = 1 - PROBS["mutation"]
        else:
            m_gene = PROBS["mutation"]
        f_gene = 1
        if father in one_gene:
            f_gene = 0.5
        elif father in two_genes:
            f_gene = 1 - PROBS["mutation"]
        else:
            f_gene = PROBS["mutation"]
        """
        mother and not father + father and not mother
        0.01 * 0.01 + (1 - 0.01) * 0.99 
        """
        if gene == 2: # 100% inheritence
            prob = m_gene * f_gene
        elif gene == 1: # Either father or mother
            prob = m_gene * (1 - f_gene) + (1 - m_gene) * f_gene
        else: # Neither father nor mother
            prob = (1 - m_gene) * (1 - f_gene)
        return prob * PROBS["trait"][gene][person in have_trait]

def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
{
  'Harry': {'name': 'Harry', 'mother': 'Lily', 'father': 'James', 'trait': None},
  'James': {'name': 'James', 'mother': None, 'father': None, 'trait': True},
  'Lily': {'name': 'Lily', 'mother': None, 'father': None, 'trait': False}
}
We will here show the calculation of joint_probability(people, {"Harry"}, {"James"}, {"James"}). Based on the arguments, one_gene is {"Harry"}, two_genes is {"James"}, and have_trait is {"James"}. 
This therefore represents the probability that: Lily has 0 copies of the gene and does not have the trait, Harry has 1 copy of the gene and does not have the trait, and James has 2 copies of the gene and does have the trait.
    """
    result = 1
    for p in people:
        #result.append(probability(people, one_gene, two_genes, have_trait, p))
        result *= probability(people, one_gene, two_genes, have_trait, p)
    print(f"joint_probability result: {result}")
    #return reduce(mul, result)
    return result

def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in probabilities:
        gene = (2 if person in two_genes else 1 if person in one_gene else 0)
        probabilities[person]["trait"][person in have_trait] += p
        probabilities[person]["gene"][gene] += p

def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    print(f"probabilities: {probabilities}")
    for k,v in probabilities.items():
        genes = sum(v["gene"].values())
        traits = sum(v["trait"].values())
        for g in v["gene"]:
            v["gene"][g] /= genes
        for g in v["trait"]:
            v["trait"][g] /= traits

"""
Only executes when this file is run as a script
Does NOT execute when this file is imported as a module
__name__ stores the name of a module when it is loaded. It is set to either the string of "__main__" if it's in the top-level or the module's name if it is being imported.
"""
if __name__ == "__main__":
    main()