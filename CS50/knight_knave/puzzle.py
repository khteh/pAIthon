from logic import *
import termcolor
from termcolor import colored, cprint
"""
Model: Assignment of a truth value to every propositional symbol (a "possible world")
Example:
P: It is raining
Q: It is Tuesday
Model: {P: true, Q: false} <= One of the possible worlds: It is raining and it is NOT Tuesday

Entailment: In every model in which sentence a is true, sentence b is also true

Inference: Uses emtailment at the core of it. Does KB entail the query is true?
One of the methods is Model Checking:
To determine if KB entails query:
- Enumerate all possible models
- If in every model where KB is true, query is true, then KB entails query.
- Otherwise KB does NOT entail the query

Disjunction: Propositional logic symbols connected using OR
Conjunction: Propositional logic symbols connected using AND
Clause: A disjunction of literals. P v Q v R
Conjunctive Normal Form: Logical sentence that is conjunction of clauses. (A v B v C) ^ (P v Q v R) ^ (Not(D) v E)
Conversion to CNF:
(1) Eliminate Biconditionals: Turn (a <-> b) into (a -> b) ^ (b -> a)
(2) Eliminate Implications: Turn (a -> b) into (Not(a) v b)
(3) Move NOT inwards using De Morgan's Laws. E.g., turn No(a ^ b) into (Not(a) v Not(b))
(4) Use distributive law to distribute 'v' wherever possible
Example: (P v Q) -> R
Not(P v Q) v R (Eliminate Implication)
(Not(P) ^ Not(Q)) v R (De Morgan's Law)
(Not(P) v R) ^ (Not(Q) v R) (Distributive law)

Objective of converting to CNF is so that we could apply Inference by Resolution to obtain/derive new information.
Example: 
(1) P v Q, Not(P) v R results in Q v R
(2) P ^ Not(P) results in () - empty clause = FALSE

Inference by Resolution
- To determine if KB entails query
  - Check if KB ^ Not(query) is FALSE / contradictory.
    - If Yes, then KB entails query
    - Otherwise, no entailment

  - Convert KB ^ Not(query) to CNF
  - Keep checking if we can use  resolution to produce a new clause
    - If this produces an empty clause (FALSE), we have a contradiction and therefore, KB entails query
    - Otherwise, if we can't add new clauses, there is NO entailment.
Example:
Does (A v B) ^ (Not(B) v C) ^ (Not(C)) entail A? Start by assuming the Not(A)
(A v B) ^ (Not(B) v C) ^ (Not(C)) ^ (Not(A))
(A v B) (Not(B) v C) (Not(C)) (Not(A)) generates (Not(B))
(A v B) (Not(A)) (Not(B)) generates (A)
(A) (Not(A)) generates () = FALSE

$ python puzzle.py
$ check50 --local ai50/projects/2024/x/knights
"""
AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

CommonKnowledge = And(
    Or(AKnight, AKnave),
    Not(And(AKnight, AKnave)),
    Or(BKnight, BKnave),
    Not(And(BKnight, BKnave)),
    Or(CKnight, CKnave),
    Not(And(CKnight, CKnave)),
)
# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    # TODO
    CommonKnowledge,
    Implication(AKnight, And(AKnight, AKnave)),
    Implication(AKnave, Not(And(AKnight, AKnave)))
)
#print(knowledge0.formula())
# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    # TODO
    CommonKnowledge,
    Implication(AKnight, And(AKnave, BKnave)),
    Implication(AKnave, Not(And(AKnave, BKnave)))
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(
    # TODO
    CommonKnowledge,
    Implication(AKnight, Or(And(AKnight, BKnight), And(AKnave, BKnave))),
    Implication(AKnave, Or(And(AKnight, BKnave), And(AKnave, BKnight))),
    Implication(BKnight, Or(And(AKnight, BKnave), And(AKnave, BKnight))),
    Implication(BKnave, Or(And(AKnight, BKnight), And(AKnave, BKnave)))
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
knowledge3 = And(
    # TODO
    CommonKnowledge,
    Implication(AKnight, Or(AKnight, AKnave)),
    Implication(AKnave, Not(Or(AKnight, AKnave))),

    Implication(BKnight, Implication(AKnight, BKnave)),
    Implication(BKnight, Implication(AKnave, BKnight)),
    Implication(BKnight, Not(Implication(AKnight, BKnave))),
    Implication(BKnight, Not(Implication(AKnave, BKnight))),

    Implication(BKnight, CKnave),
    Implication(BKnave, CKnight),
    Implication(CKnight, AKnight),
    Implication(CKnave, CKnave)
)

def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    termcolor.cprint(f"    {symbol}: YES", "green")
                elif not model_check(knowledge, Not(symbol)):
                    print(f"    {symbol}: MAYBE")

"""
Only executes when this file is run as a script
Does NOT execute when this file is imported as a module
__name__ stores the name of a module when it is loaded. It is set to either the string of "__main__" if it's in the top-level or the module's name if it is being imported.
"""
if __name__ == "__main__":
    main()
