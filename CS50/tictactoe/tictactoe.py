"""
Tic Tac Toe Player
"""

import math
from math import inf
from copy import copy, deepcopy
X = "X"
O = "O"
EMPTY = None

def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    N = len(board)
    x = 0
    o = 0
    for i in range(0,N):
        for j in range(0, N):
            if board[i][j] == X:
                x += 1
            elif board[i][j] == O:
                o += 1
    if x == 0 or x <= o:
        return X
    return O

def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    N = len(board)
    result = set()
    for i in range(0,N):
        for j in range(0, N):
            if board[i][j] == EMPTY:
                result.add((i,j))
    return result

def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    turn = player(board)
    newBoard = deepcopy(board)
    N = len(newBoard)
    if action[0] >= 0 and action[0] < N and action[1] >= 0 and action[1] < N and newBoard[action[0]][action[1]] == EMPTY:
        if turn == X:
            newBoard[action[0]][action[1]] = X
        else:
            newBoard[action[0]][action[1]] = O
    else:
        raise RuntimeError(f"Invalid move!")
    return newBoard

def primary_diagonal(matrix):
    sum = 0
    for i in range(len(matrix)):
        sum += matrix[i][i]
    return sum

def secondary_left_diagonal(matrix):
    sum = 0
    for i in range(len(matrix)):
        sum += matrix[i][len(matrix) - i - 1]
    return sum

def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    N = len(board)
    state = [[0]*N,
            [0]*N,
            [0]*N]
    for i in range(0,N):
        for j in range(0, N):
            if board[i][j] == X:
                state[i][j] = 1
            elif board[i][j] == O:
                state[i][j] = -1
    row_totals = [ sum(x) for x in state ]
    col_totals = [ sum(x) for x in zip(*state) ]
    diag1 = primary_diagonal(state)
    diag2 = secondary_left_diagonal(state)
    if N in row_totals or N in col_totals or diag1 == N or diag2 == N:
        return X
    elif -N in row_totals or -N in col_totals or diag1 == -N or diag2 == -N:
        return O
    return None

def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    w = winner(board)
    if w != None:
        return True
    N = len(board)
    for i in range(0,N):
        for j in range(0, N):
            if board[i][j] == EMPTY:
                return False
    return True

def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    w = winner(board)
    if w == X:
        return 1
    elif w == O:
        return -1
    return 0

def MinValue(board):
    N = len(board)
    if terminal(board):
        return (utility(board), None)
    value = inf
    r = (value, None)
    for action in actions(board):
        v = min(value, MaxValue(result(board, action))[0])
        if v < value:
            value = v
            r = (v, action)
    return r

def MaxValue(board):
    N = len(board)
    if terminal(board):
        return (utility(board), None)
    value = -inf
    r = (value, None)
    for action in actions(board):
        v = max(value, MinValue(result(board, action))[0])
        if v > value:
            value = v
            r = v, action
    return r

def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    MAX picks action a in ACTIONS(s) that produces the highest value of MIN-VALUE(RESULT(s,a))
    MIN picks action a in ACTIONS(s) that produces the smallest value of MAX-VALUE(RESULT(s,a))
    """
    if terminal(board):
        return None
    turn = player(board)
    if turn == X:
        return MaxValue(board)[1]
    else:
        return MinValue(board)[1]