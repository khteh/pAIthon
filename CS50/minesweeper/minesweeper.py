import itertools, random
from numpy.random import Generator, PCG64DXSM
rng = Generator(PCG64DXSM())

"""
$ pipenv run python -m runner
$ pipenv run check50 --local ai50/projects/2024/x/minesweeper
"""
class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.

    In general, we’ll only want our sentences to be about cells that are not yet known to be either safe or mines. 
    This means that, once we know whether a cell is a mine or not, we can update our sentences to simplify them and potentially draw new conclusions.    
    """
    def __init__(self, cells, count):
        if count < 0:
            raise Exception(f"invalid count! {count}")
        if len(cells) == 0:
            raise Exception(f"invalid empty cells!")
        print(f"Sentence(): {cells} {count}")
        self.cells = set(cells)
        self.count = count
        self.mines = set()
        self.safes = set()
        if len(cells) == count:
            self.mines = set(cells)
        elif count == 0:
            self.safes = set(cells)

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        return self.mines

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        return self.safes

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            print(f"Sentence mark_mine {cell}...", end='')
            self.cells.remove(cell)
            self.mines.add(cell)
            self.count -= 1
            print(f"{self.cells}, {self.mines}, {self.count}")

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            print(f"Sentence mark_safe {cell}...", end='')
            self.cells.remove(cell)
            self.safes.add(cell)
            if len(self.cells) == 0:
                self.count = 0
            print(f"{self.cells}, {self.safes}, {self.count}")

class MinesweeperAI():
    """
    Minesweeper game player
    """
    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        print(f"MinesweeperAI mark_mine {cell}")
        self.mines.add(cell)
        if cell in self.safes:
            self.safes.remove(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        print(f"MinesweeperAI mark_safe {cell}")
        self.safes.add(cell)
        if cell in self.mines:
            self.mines.remove(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        print(f"MinesweeperAI add_knowledge cell: {cell}, count: {count}")
        # 1) mark the cell as a move that has been made
        self.moves_made.add(cell)
        # 2) mark the cell as safe
        self.mark_safe(cell)
        """
        3) add a new sentence to the AI's knowledge base
            based on the value of `cell` and `count`
        """
        cells = set()
        for i in range(max(0, cell[0]-1), min(self.height, cell[0]+2)):
            for j in range(max(0, cell[1]-1), min(self.width, cell[1]+2)):
                if (i,j) != cell and (i,j) not in self.moves_made and (i,j) not in self.mines and (i,j) not in self.safes:
                    cells.add((i,j))
                elif (i,j) in self.mines and count > 0:
                    count -= 1
        if len(cells):
            sentence = Sentence(cells, count)
            self.knowledge.append(sentence)
        """
        4) mark any additional cells as safe or as mines
            if it can be concluded based on the AI's knowledge base
        """
        for sentence in self.knowledge:
            safes = sentence.known_safes().copy()
            mines = sentence.known_mines().copy()
            for safe in safes:
                self.mark_safe(safe)
            for mine in mines:
                self.mark_mine(mine)
        """
        5) add any new sentences to the AI's knowledge base
            if they can be inferred from existing knowledge
        """
        knowledge = []
        for s1 in self.knowledge:
            for s2 in self.knowledge:
                cells = set()
                count = 0
                if s1 != s2:
                    cells = s2.cells - s1.cells if s1.cells.issubset(s2.cells) else s1.cells - s2.cells
                    count = s2.count - s1.count if s1.cells.issubset(s2.cells) else s1.count - s2.count
                if len(cells) > 0 and count > 0:
                    s = Sentence(cells, count)
                    if s not in knowledge and s not in self.knowledge:
                        knowledge.append(s)
        self.knowledge.extend(knowledge)
        """
        Go through all sentences and mark cells as safe or mines
        If sentence.count == 0, mark all cells as safe
        If sentence.count == len(sentence.cells), mark all cells as mines        
        """
        for sentence in self.knowledge:
            if sentence.count == 0:
                for c in sentence.cells.copy():
                    self.mark_safe(c)
            elif sentence.count == len(sentence.cells):
                for c in sentence.cells.copy():
                    self.mark_mine(c)

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        """
        for i in range(0, self.height):
            for j in range(0, self.width):
                if (i,j) not in self.mines and (i,j) not in self.moves_made:
                    return (i,j)
        """
        if len(self.safes):
            move = self.safes.pop()
            if move not in self.moves_made:
                return move
        return None
    
    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        i = random.randrange(self.height)
        j = random.randrange(self.width)
        if (i,j) not in self.mines and (i,j) not in self.moves_made:
            return (i,j)
        """
        all_cells = set(itertools.product(range(self.height), range(self.width)))
        available_moves = list(all_cells - self.moves_made - self.mines)
        if available_moves:
            return rng.choice(available_moves)
        return None