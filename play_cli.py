from collections import defaultdict
from logging import getLogger

import cchess_alphazero.environment.static_env as senv
from cchess_alphazero.environment.chessboard import Chessboard
from cchess_alphazero.environment.chessman import *
from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.agent.player import CChessPlayer, VisitState
from cchess_alphazero.agent.api import CChessModelAPI
from cchess_alphazero.config import Config
from cchess_alphazero.environment.env import CChessEnv
from cchess_alphazero.environment.lookup_tables import Winner, ActionLabelsRed, flip_move
from cchess_alphazero.lib.model_helper import load_best_model_weight
from cchess_alphazero.lib.tf_util import set_session_config

logger = getLogger(__name__)

def start(config: Config, human_move_first=True):
    """
    启动命令行人机对弈
    :param config: 配置对象
    :param human_move_first: 是否人类先手
    """
    set_session_config(per_process_gpu_memory_fraction=1, allow_growth=True, device_list=config.opts.device_list)
    play = PlayWithHuman(config)
    play.start(human_move_first)

class PlayWithHuman:
    """
    命令行人机对弈主类
    """
    def __init__(self, config: Config):
        self.config = config
        self.env = CChessEnv()
        self.model = None
        self.pipe = None
        self.ai = None
        self.chessmans = None
        self.human_move_first = True
        # 棋局历史，保存每步后的状态
        self.history = []

    def load_model(self):
        """
        加载AI模型和最佳权重
        """
        self.model = CChessModel(self.config)
        if self.config.opts.new or not load_best_model_weight(self.model):
            self.model.build()

    def undo_move(self):
        """
        悔棋功能：回退到上一步（建议悔棋时回退两步：人和AI各一步）
        """
        if len(self.history) > 2:
            self.history.pop()  # 弹出当前状态
            self.history.pop()  # 再弹出AI上一步
            prev_state = self.history[-1]
            self.env.set_state(prev_state)
            self.env.board.print_to_cl()
            print("悔棋成功，已回到上一步（人和AI各一步）。")
        elif len(self.history) > 1:
            # 只剩初始和一步，退回初始
            self.history.pop()
            prev_state = self.history[-1]
            self.env.set_state(prev_state)
            self.env.board.print_to_cl()
            print("悔棋成功，已回到初始。")
        else:
            print("已经是初始局面，无法悔棋。")

    def start(self, human_first=True):
        """
        主循环，控制人机对弈流程
        :param human_first: 是否人类先手
        """
        self.env.reset()
        self.load_model()
        self.pipe = self.model.get_pipes()
        self.ai = CChessPlayer(self.config, search_tree=defaultdict(VisitState), pipes=self.pipe,
                              enable_resign=True, debugging=False)
        self.human_move_first = human_first

        labels = ActionLabelsRed
        labels_n = len(ActionLabelsRed)

        # 记录初始状态
        self.history = [self.env.get_state()]

        self.env.board.print_to_cl()

        while not self.env.board.is_end():
            # 人类走棋
            if human_first == self.env.red_to_move:
                self.env.board.calc_chessmans_moving_list()
                is_correct_chessman = False
                is_correct_position = False
                chessman = None

                # 循环直到选对棋子或悔棋
                while not is_correct_chessman:
                    title = "请输入棋子位置(或输入undo悔棋): "
                    input_chessman_pos = input(title).strip()
                    if input_chessman_pos.lower() == "undo":
                        self.undo_move()
                        continue
                    if len(input_chessman_pos) < 2 or not input_chessman_pos.isdigit():
                        print("输入无效，请输入如 23 这样的坐标")
                        continue
                    x, y = int(input_chessman_pos[0]), int(input_chessman_pos[1])
                    chessman = self.env.board.chessmans[x][y]
                    if chessman != None and chessman.is_red == self.env.board.is_red_turn:
                        is_correct_chessman = True
                        print(f"当前棋子为{chessman.name_cn}，可以落子的位置有：")
                        for point in chessman.moving_list:
                            print(point.x, point.y)
                    else:
                        print("没有找到此名字的棋子或未轮到此方走子")
                # 循环直到落子合法或悔棋
                while not is_correct_position:
                    title = "请输入落子的位置(或输入undo悔棋): "
                    input_chessman_pos = input(title).strip()
                    if input_chessman_pos.lower() == "undo":
                        self.undo_move()
                        is_correct_chessman = False
                        break
                    if len(input_chessman_pos) < 2 or not input_chessman_pos.isdigit():
                        print("输入无效，请输入如 23 这样的坐标")
                        continue
                    x, y = int(input_chessman_pos[0]), int(input_chessman_pos[1])
                    is_correct_position = chessman.move(x, y)
                    if is_correct_position:
                        # 记录本步后的状态
                        self.history.append(self.env.get_state())
                        self.env.board.print_to_cl()
                        self.env.board.clear_chessmans_moving_list()
            else:
                # AI走棋
                action, policy = self.ai.action(self.env.get_state(), self.env.num_halfmoves)
                if not self.env.red_to_move:
                    action = flip_move(action)
                if action is None:
                    print("AI投降了!")
                    break
                self.env.step(action)
                print(f"AI选择移动 {action}")
                # 记录本步后的状态
                self.history.append(self.env.get_state())
                self.env.board.print_to_cl()

        self.ai.close()
        print(f"胜者是 {self.env.board.winner} !!!")
        self.env.board.print_record()