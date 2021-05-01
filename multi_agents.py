import numpy as np
import abc
import util
from game import Agent, Action


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.
        get_action chooses among the best options according to the evaluation function.
        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.
        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.
        """

        # Useful information you can extract from a GameState (game_state.py)

        successor_game_state = current_game_state.generate_successor(action=action)
        max_tile = successor_game_state.max_tile
        score = successor_game_state.score
        empty_tiles = len(successor_game_state.get_empty_tiles()[0])

        #  A linear combination of the current max_tile, the score and number of empty_tiles,
        #  a higher score and higher max_tile value are good, and the more empty_tiles there are
        #  the better the situation the player is in because he has more moves to make.
        return max_tile + score + (empty_tiles * 50)


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.
    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.
    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.
    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


class MinmaxAgent(MultiAgentSearchAgent):

    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        Here are some method calls that might be useful when implementing minimax.
        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1
        Action.STOP:
            The stop direction, which is always legal
        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        legal_actions = game_state.get_legal_actions(0)
        best_score = -np.inf
        best_action = None
        for action in legal_actions:  # go over legal actions to find best minimax one
            depth = 0
            cur_score = self.min_value(game_state.generate_successor(0, action), depth)  # get minimax value for action
            if cur_score > best_score:
                best_action = action
                best_score = cur_score
        return best_action

    def max_value(self, state, depth):
        """
        Max value evaluation as part of minimax algorithm
        :param state: current game state
        :param depth: current depth
        :return: max possible value in respect to minimax
        """
        if state.done or depth == self.depth:  # terminal node or at max depth
            return self.evaluation_function(state)
        legal_actions = state.get_legal_actions(0)
        best_score = -np.inf
        for action in legal_actions:
            best_score = max(best_score, self.min_value(state.generate_successor(0, action), depth))
        return best_score

    def min_value(self, state, depth):
        """
        Min value evaluation as part of minimax algorithm
        :param state: current game state
        :param depth: current depth
        :return: min possible value in respect to minimax
        """
        depth = depth + 1  # single play is one agent move and one board move, so increment depth here
        if state.done:
            return self.evaluation_function(state)
        legal_actions = state.get_legal_actions(1)
        best_score = np.inf
        for action in legal_actions:
            best_score = min(best_score, self.max_value(state.generate_successor(1, action), depth))
        return best_score


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        legal_actions = game_state.get_legal_actions(0)
        best_score = -np.inf
        best_action = None
        for action in legal_actions:
            cur_score = self.alpha_beta(game_state.generate_successor(0, action), self.depth, -np.inf, np.inf, 1)
            if cur_score > best_score:
                best_action = action
                best_score = cur_score
        return best_action

    def alpha_beta(self, game_state, depth, a, b, player):
        """
        Alpha beta pruning algorithm
        :param game_state: current game state
        :param depth: current depth
        :param a: alpha value
        :param b: beta value
        :param player: 0 for max player, 1 for min player
        :return: minimax value in respect to depth
        """
        if game_state.done or depth == 0:
            return self.evaluation_function(game_state)
        if player == 1:
            depth -= 1
        legal_actions = game_state.get_legal_actions(player)
        for action in legal_actions:
            if player == 0:
                a = max(a, self.alpha_beta(game_state.generate_successor(player, action), depth, a, b, 1 - player))
            else:
                b = min(b, self.alpha_beta(game_state.generate_successor(player, action), depth, a, b, 1 - player))
            if b <= a:
                break
        if player == 0:
            return a
        if player == 1:
            return b


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        legal_actions = game_state.get_legal_actions(0)
        best_score = -np.inf
        best_action = None
        for action in legal_actions:
            depth = 0
            cur_score = self.exp_value(game_state.generate_successor(0, action), depth)
            if cur_score > best_score:
                best_action = action
                best_score = cur_score
        return best_action

    def max_value(self, state, depth):
        """
        Max value evaluation as part of expectimax algorithm
        :param state: current game state
        :param depth: current depth
        :return: max possible value in respect to expectimax
        """
        if state.done or depth == self.depth:
            return self.evaluation_function(state)
        legal_actions = state.get_legal_actions(0)
        best_score = -np.inf
        for action in legal_actions:
            best_score = max(best_score, self.exp_value(state.generate_successor(0, action), depth))
        return best_score

    def exp_value(self, state, depth):
        """
        Expected value node evaluation as part of expectimax algorithm
        :param state: current game state
        :param depth: current depth
        :return: expected value in expected value nodes
        """
        depth = depth + 1
        if state.done:
            return self.evaluation_function(state)
        legal_actions = state.get_legal_actions(1)
        scores_sum = 0
        for action in legal_actions:
            scores_sum += self.max_value(state.generate_successor(1, action), depth)
        return scores_sum / len(legal_actions)


def get_num_adjacent(board):
    """
    Gets the number of equal adjacent tiles in board and the possible value of adding them
    :param board: the current game board
    """
    num = 0
    for i in range(len(board)):
        for j in range(len(board)):
            if i > 0:
                if board[i][j] == board[i - 1][j]:
                    num += board[i][j] * 2
            if i < len(board) - 1:
                if board[i][j] == board[i + 1][j]:
                    num += board[i][j] * 2
            if j > 0:
                if board[i][j] == board[i][j - 1]:
                    num += board[i][j] * 2
            if j < len(board) - 1:
                if board[i][j] == board[i][j + 1]:
                    num += board[i][j] * 2
    return num


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).
    DESCRIPTION: This evaluation function is a linear combination of the max_tile value, the score
                 number of empty_tiles, number of equal adjacent tiles with respect to their values and number of
                 high value tiles on the edges.
                 The more empty tiles there are the more space is left on the board to play, equal adjacent tiles
                 can be merged for a higher score and so they indicate a good state. High value tiles on edges make
                 sure that when the opponent randomly places new low-valued tiles on the board, the high tiles on edges
                 that cannot be merged with the new low-valued tiles won't conflict.
                 The weights of the linear combination came from trial and error and we feel that these weights gave the
                 best average results.
    """
    board = current_game_state.board
    max_tile = current_game_state.max_tile
    score = current_game_state.score
    min_idx = 0
    max_idx = 3
    big_on_edges = 0
    for i in range(min_idx, max_idx + 1):
        if board[min_idx][i] >= max_tile / 4 and max_tile >= 64:
            big_on_edges += 1
        if board[max_idx][i] >= max_tile / 4 and max_tile >= 64:
            big_on_edges += 1
        if 1 <= i <= 2:
            if board[i][min_idx] >= max_tile / 4 and max_tile >= 64:
                big_on_edges += 1
            if board[i][max_idx] >= max_tile / 4 and max_tile >= 64:
                big_on_edges += 1

    empty_tiles = len(current_game_state.get_empty_tiles()[0])
    adjacent = get_num_adjacent(board)

    return max_tile + (big_on_edges * max_tile / 2) + score + (empty_tiles * 50) + adjacent


# Abbreviation
better = better_evaluation_function
