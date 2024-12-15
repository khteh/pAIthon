import sys
from crossword import *
from itertools import permutations
from pprint import pprint
"""
Constraint Satisfaction Problem:
- Set of variables {x1, x2, x3, ...}
- Set of domains for each variable {d1, d2, d3, ...}
- Set of constraints C
  - Hard constraints: Must be met
  - Soft constraints: Express some notion of which soluhtions are preferred over others
  - Unary constraints: Involve single variable
  - Binary constraints: Involve 2 variables
Node consistency: All the values in a variable's domain satisfy the variable's unary constraints
Arc consistency: All the values in a variable's domain satisfy the variable's binary constraints
                 To make X arc-consistent with respect to Y, remove elements from X's domain until every choice for X has a possible choice for Y

Python has a library called "python-constraint" for this problem

$ pipenv install Pillow
$ python generate.py data/structure0.txt data/words0.txt
$ check50 --local ai50/projects/2024/x/crossword
"""
class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )
        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Node consistency
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        #print(f"domains: {self.domains}")
        for k,v in self.domains.items():
            words = set()
            for str in v:
                print(f"str: {str}, ", end="")
                if len(str) == k.length:
                    words.add(str)
                #print(f"words: {words}")
            #print()
            self.domains[k] = words
    def revise(self, x, y):
        """
        ARC consistency
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revised = False
        # For any pair of variables v1, v2, their overlap is either:
        #    None, if the two variables do not overlap; or
        #    (i, j), where v1's ith character overlaps v2's jth character
        #print(f"x: {x}, y: {y}")
        overlap = self.crossword.overlaps[x, y]
        if overlap:
            i = overlap[0]
            j = overlap[1]
            #print(f"i: {i}, j: {j}")
            words = self.domains[x].copy()
            for str1 in words:
                flag = False
                for str2 in self.domains[y]:
                    #print(f"str1: {str1}, str2: {str2}")
                    if len(str1) > i and len(str2) > j and str1[i] == str2[j]:
                        flag = True
                        break
                if not flag:
                    revised = True
                    self.domains[x].remove(str1)
        return revised
    
    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        if arcs == None:
            arcs = []
            for i, j in permutations(list(self.domains.keys()), 2):
                arcs.append((i,j))
        while arcs:
            variable = arcs[0]
            del arcs[0]
            if self.revise(variable[0], variable[1]):
                if not self.domains[variable[0]] or len(self.domains[variable[0]]) == 0:
                    return False
                for v in self.crossword.neighbors(variable[0]):
                    if v != variable[0] and v != variable[1]:
                        arcs.append((v, variable[0]))
        return True
    
    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        for k,v in assignment.items():
            if not v or len(v) == 0:
                return False
        for k,v in self.domains.items():
            if k not in assignment:
                return False
        return True
    
    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        An assignment is consistent if it satisfies all of the constraints of the problem: that is to say, all values are distinct, every value is the correct length, and there are no conflicts between neighboring variables.
        """
        unique = set()
        for k,v in assignment.items():
            if len(unique) > 0 and v in unique:
                return False
            unique.add(v)
            if len(v) != k.length:
                return False
            neighbours = set(
                v for v in assignment
                if v != k and self.crossword.overlaps[v, k]
            )
            for v1 in neighbours:
                overlap = self.crossword.overlaps[k, v1]
                i = overlap[0]
                j = overlap[1]
                str = assignment[v1]
                if len(v) > i and len(str) > j and v[i] != str[j]:
                    return False
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.

        The least-constraining values heuristic is computed as the number of values ruled out for neighboring unassigned variables. That is to say, if assigning var to a particular value results in eliminating n possible choices for neighboring variables, you should order your results in ascending order of n.
        """
        result = {key: 0 for key in self.domains[var]}
        for str1 in self.domains[var]:
            for v in self.crossword.neighbors(var):
                if v not in assignment:
                    overlap = self.crossword.overlaps[var, v]
                    if overlap:
                        i = overlap[0]
                        j = overlap[1]
                        for str2 in self.domains[v]:
                            if len(str1) > i and len(str2) > j and str1[i] != str2[j]:
                             result[str1] += 1
        result = {k: v for k, v in sorted(result.items(), key=lambda item: item[1])}
        return list(result.keys())

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        result = dict()
        for k,v in self.domains.items():
            if k not in assignment:
                neighbours = len(self.crossword.neighbors(k))
                result[k] = (len(v), neighbours)
        result = {k: v for k, v in sorted(result.items(), key=lambda item: (item[1][0], -item[1][1]))}
        return list(result.keys())[0] if len(result) > 0 else None
    
    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if self.assignment_complete(assignment):
            return assignment
        v = self.select_unassigned_variable(assignment)
        for s in self.domains[v]:
            assignment[v] = s
            if self.consistent(assignment) and self.backtrack(assignment):
                return assignment
            assignment.remove(v)
        return None
    
def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    #creator.enforce_node_consistency()
    #creator.ac3()
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)
if __name__ == "__main__":
    main()
