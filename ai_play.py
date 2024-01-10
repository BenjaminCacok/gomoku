from __future__ import print_function
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet


def run():
    n = 5
    width, height = 8, 8
    # change the AI model by editing the model_file
    model_file = 'best_policy.model'
    try:
        board = Board(width=8, height=8, n_in_row=5)
        game = Game(board)

        best_policy = PolicyValueNet(width, height, model_file=model_file)
        mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                                 c_puct=5,
                                 n_playout=400)

        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=2000)

        game.start_play_presentation(pure_mcts_player, mcts_player, start_player=0, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run()
