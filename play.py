import sys
import pygame
import random
import os.path
import time
import copy
import numpy as np

from pygame.locals import *
from logging import getLogger
from collections import defaultdict
from threading import Thread
from time import sleep
from datetime import datetime

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
main_dir = os.path.split(os.path.abspath(__file__))[0]
PIECE_STYLE = 'WOOD'


def start(config: Config, human_move_first=True):
    """
    启动人机对弈界面，human_move_first控制人机先后手。
    """
    global PIECE_STYLE
    PIECE_STYLE = config.opts.piece_style
    play = PlayWithHuman(config)
    play.start(human_move_first)


class PlayWithHuman:
    """
    人机对弈主界面与流程控制类（基于pygame图形界面）。
    """

    def __init__(self, config: Config):
        self.config = config
        self.env = CChessEnv()
        self.model = None
        self.pipe = None
        self.ai = None
        self.winstyle = 0
        self.chessmans = None
        self.human_move_first = True
        self.screen_width = 720
        self.height = 577
        self.width = 521
        self.chessman_w = 57
        self.chessman_h = 57
        self.disp_record_num = 15
        self.rec_labels = [None] * self.disp_record_num
        self.nn_value = 0
        self.mcts_moves = {}
        # 历史保存为状态链表，支持悔棋
        self.history = []
        if self.config.opts.bg_style == 'WOOD':
            self.chessman_w += 1
            self.chessman_h += 1
        # AI线程管理
        self.ai_should_exit = False
        self.ai_worker = None

    def load_model(self):
        # 加载模型及权重，如果无则新建
        self.model = CChessModel(self.config)
        if self.config.opts.new or not load_best_model_weight(self.model):
            self.model.build()

    def init_screen(self):
        """
        初始化pygame窗口、棋盘和右侧信息区。
        """
        bestdepth = pygame.display.mode_ok([self.screen_width, self.height], self.winstyle, 32)
        screen = pygame.display.set_mode([self.screen_width, self.height], self.winstyle, bestdepth)
        pygame.display.set_caption("中国象棋Zero")
        # 背景贴图
        bgdtile = load_image(f'{self.config.opts.bg_style}.GIF')
        bgdtile = pygame.transform.scale(bgdtile, (self.width, self.height))
        board_background = pygame.Surface([self.width, self.height])
        board_background.blit(bgdtile, (0, 0))
        widget_background = pygame.Surface([self.screen_width - self.width, self.height])
        white_rect = Rect(0, 0, self.screen_width - self.width, self.height)
        widget_background.fill((255, 255, 255), white_rect)

        # 文字标签
        font_file = self.config.resource.font_path
        font = pygame.font.Font(font_file, 16)
        font_color = (0, 0, 0)
        font_background = (255, 255, 255)
        t = font.render("着法记录", True, font_color, font_background)
        t_rect = t.get_rect()
        t_rect.x = 10
        t_rect.y = 10
        widget_background.blit(t, t_rect)

        # 不再绘制悔棋按钮区域

        screen.blit(board_background, (0, 0))
        screen.blit(widget_background, (self.width, 0))
        pygame.display.flip()
        self.chessmans = pygame.sprite.Group()
        creat_sprite_group(self.chessmans, self.env.board.chessmans_hash, self.chessman_w, self.chessman_h)
        return screen, board_background, widget_background

    def start(self, human_first=True):
        """
        主循环，处理事件、刷新界面、调度AI线程。human_first控制人机先后手。
        """
        self.env.reset()
        self.load_model()
        self.pipe = self.model.get_pipes()
        self.ai = CChessPlayer(self.config, search_tree=defaultdict(VisitState), pipes=self.pipe,
                               enable_resign=True, debugging=True)
        self.human_move_first = human_first

        pygame.init()
        screen, board_background, widget_background = self.init_screen()
        framerate = pygame.time.Clock()

        labels = ActionLabelsRed
        labels_n = len(ActionLabelsRed)

        current_chessman = None
        if human_first:
            self.env.board.calc_chessmans_moving_list()

        # 历史初始化为初始状态（保存FEN、着法记录、着法编号）
        self.history = [(self.env.observation, self.env.board.record, self.env.board.turns)]

        # 启动AI线程
        self.ai_should_exit = False
        self.ai_worker = Thread(target=self.ai_move, name="ai_worker")
        self.ai_worker.daemon = True
        self.ai_worker.start()

        while not self.env.board.is_end():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.board.print_record()
                    self.ai_should_exit = True
                    if self.ai_worker:
                        self.ai_worker.join(timeout=2)
                    game_id = datetime.now().strftime("%Y%m%d-%H%M%S")
                    path = os.path.join(self.config.resource.play_record_dir,
                                        self.config.resource.play_record_filename_tmpl % game_id)
                    self.env.board.save_record(path)
                    sys.exit()
                elif event.type == VIDEORESIZE:
                    pass
                elif event.type == KEYDOWN:
                    # 按下'u'键悔棋
                    if event.key == pygame.K_u:
                        self.undo_move()
                        # 重新创建棋子精灵组
                        self.chessmans.empty()
                        creat_sprite_group(self.chessmans, self.env.board.chessmans_hash, self.chessman_w,
                                           self.chessman_h)
                        current_chessman = None
                        continue
                elif event.type == MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    # 不再检查悔棋按钮区域
                    if human_first == self.env.red_to_move:
                        pressed_array = pygame.mouse.get_pressed()
                        for index in range(len(pressed_array)):
                            if index == 0 and pressed_array[index]:
                                col_num, row_num = translate_hit_area(mouse_x, mouse_y, self.chessman_w,
                                                                      self.chessman_h)
                                chessman_sprite = select_sprite_from_group(
                                    self.chessmans, col_num, row_num)
                                if current_chessman is None and chessman_sprite != None:
                                    if chessman_sprite.chessman.is_red == self.env.red_to_move:
                                        current_chessman = chessman_sprite
                                        chessman_sprite.is_selected = True
                                elif current_chessman != None and chessman_sprite != None:
                                    if chessman_sprite.chessman.is_red == self.env.red_to_move:
                                        current_chessman.is_selected = False
                                        current_chessman = chessman_sprite
                                        chessman_sprite.is_selected = True
                                    else:
                                        move = str(current_chessman.chessman.col_num) + str(
                                            current_chessman.chessman.row_num) + \
                                               str(col_num) + str(row_num)
                                        success = current_chessman.move(col_num, row_num, self.chessman_w,
                                                                        self.chessman_h)
                                        # 记录落子后棋盘状态、着法记录、编号
                                        if success:
                                            self.chessmans.remove(chessman_sprite)
                                            chessman_sprite.kill()
                                            current_chessman.is_selected = False
                                            current_chessman = None
                                            self.history.append(
                                                (self.env.observation, self.env.board.record, self.env.board.turns))
                                elif current_chessman != None and chessman_sprite is None:
                                    move = str(current_chessman.chessman.col_num) + str(
                                        current_chessman.chessman.row_num) + \
                                           str(col_num) + str(row_num)
                                    success = current_chessman.move(col_num, row_num, self.chessman_w, self.chessman_h)
                                    # 记录落子后棋盘状态、着法记录、编号
                                    if success:
                                        current_chessman.is_selected = False
                                        current_chessman = None
                                        self.history.append(
                                            (self.env.observation, self.env.board.record, self.env.board.turns))

            self.draw_widget(screen, widget_background)
            framerate.tick(20)
            # 清空上一次的精灵
            self.chessmans.clear(screen, board_background)

            # 刷新所有棋子精灵
            self.chessmans.update()
            self.chessmans.draw(screen)
            pygame.display.update()

        self.ai_should_exit = True
        if self.ai_worker:
            self.ai_worker.join(timeout=2)
        logger.info(f"Winner is {self.env.board.winner} !!!")
        self.env.board.print_record()
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(self.config.resource.play_record_dir,
                            self.config.resource.play_record_filename_tmpl % game_id)
        self.env.board.save_record(path)
        sleep(3)

    def undo_move(self):
        """
        悔棋功能：回退到上一步（建议悔棋时回退两步：人和AI各一步）
        """
        # 通知AI线程退出
        self.ai_should_exit = True
        if self.ai_worker:
            self.ai_worker.join(timeout=2)  # 等待AI线程退出，防止多线程写冲突
        if len(self.history) > 2:
            self.history.pop()
            self.history.pop()
            prev_state, prev_record, prev_turns = self.history[-1]
            self.env.set_state(prev_state)
            self.env.board.record = prev_record
            self.env.board.turns = prev_turns
            self.env.board.print_to_cl()
            print("悔棋成功，已回到上一步（人和AI各一步）。")
        elif len(self.history) > 1:
            self.history.pop()
            prev_state, prev_record, prev_turns = self.history[-1]
            self.env.set_state(prev_state)
            self.env.board.record = prev_record
            self.env.board.turns = prev_turns
            self.env.board.print_to_cl()
            print("悔棋成功，已回到初始。")
        else:
            print("已经是初始局面，无法悔棋。")

        # 重启AI线程
        self.ai_should_exit = False
        self.ai_worker = Thread(target=self.ai_move, name="ai_worker")
        self.ai_worker.daemon = True
        self.ai_worker.start()

    def ai_move(self):
        ai_move_first = not self.human_move_first
        no_act = None
        while not self.env.done and not self.ai_should_exit:
            if ai_move_first == self.env.red_to_move:
                labels = ActionLabelsRed
                labels_n = len(ActionLabelsRed)
                self.ai.search_results = {}
                state = self.env.get_state()
                logger.info(f"state = {state}")
                _, _, _, check = senv.done(state, need_check=True)
                if not check and state in [h[0] for h in self.history[:-1]]:
                    no_act = []
                    free_move = defaultdict(int)
                    for i in range(len(self.history) - 1):
                        if self.history[i][0] == state:
                            # 如果走了下一步是将军或捉：禁止走那步
                            if senv.will_check_or_catch(state, self.history[i + 1][0]):
                                no_act.append(self.history[i + 1][0])
                            # 否则当作闲着处理
                            else:
                                free_move[state] += 1
                                if free_move[state] >= 2:
                                    # 作和棋处理
                                    self.env.winner = Winner.draw
                                    self.env.board.winner = Winner.draw
                                    break
                    if no_act:
                        logger.debug(f"no_act = {no_act}")
                action, policy = self.ai.action(state, self.env.num_halfmoves, no_act)
                if action is None:
                    logger.info("AI has resigned!")
                    return
                # AI走棋后，刷新棋子精灵
                if not self.env.red_to_move:
                    action = flip_move(action)
                key = self.env.get_state()
                p, v = self.ai.debug[key]
                logger.info(f"check = {check}, NN value = {v:.3f}")
                self.nn_value = v
                logger.info("MCTS results:")
                self.mcts_moves = {}
                for move, action_state in self.ai.search_results.items():
                    move_cn = self.env.board.make_single_record(int(move[0]), int(move[1]), int(move[2]), int(move[3]))
                    logger.info(
                        f"move: {move_cn}-{move}, visit count: {action_state[0]}, Q_value: {action_state[1]:.3f}, Prior: {action_state[2]:.3f}")
                    self.mcts_moves[move_cn] = action_state
                x0, y0, x1, y1 = int(action[0]), int(action[1]), int(action[2]), int(action[3])
                chessman_sprite = select_sprite_from_group(self.chessmans, x0, y0)
                sprite_dest = select_sprite_from_group(self.chessmans, x1, y1)
                if sprite_dest:
                    self.chessmans.remove(sprite_dest)
                    sprite_dest.kill()
                if chessman_sprite:
                    chessman_sprite.move(x1, y1, self.chessman_w, self.chessman_h)
                # 记录AI落子后棋盘状态、着法记录、编号
                self.history.append((self.env.observation, self.env.board.record, self.env.board.turns))
            else:
                # 如果不是AI走棋，AI线程sleep等待下一轮
                time.sleep(0.05)

    def draw_widget(self, screen, widget_background):
        """
        绘制右侧信息区域，包括着法记录、评估等
        """
        white_rect = Rect(0, 0, self.screen_width - self.width, self.height)
        widget_background.fill((255, 255, 255), white_rect)
        pygame.draw.line(widget_background, (255, 0, 0), (10, 285), (self.screen_width - self.width - 10, 285))
        screen.blit(widget_background, (self.width, 0))
        # 不再绘制悔棋按钮
        self.draw_records(screen, widget_background)
        self.draw_evaluation(screen, widget_background)

    def draw_records(self, screen, widget_background):
        """
        绘制着法记录列表
        """
        text = '着法记录'
        self.draw_label(screen, widget_background, text, 10, 16, 10)
        records = self.env.board.record.split('\n')
        font_file = self.config.resource.font_path
        font = pygame.font.Font(font_file, 12)
        i = 0
        for record in records[-self.disp_record_num:]:
            self.rec_labels[i] = font.render(record, True, (0, 0, 0), (255, 255, 255))
            t_rect = self.rec_labels[i].get_rect()
            t_rect.y = 35 + i * 15
            t_rect.x = 10
            t_rect.width = self.screen_width - self.width
            widget_background.blit(self.rec_labels[i], t_rect)
            i += 1
        screen.blit(widget_background, (self.width, 0))

    def draw_evaluation(self, screen, widget_background):
        """
        绘制右侧NN评估和MCTS搜索信息
        """
        title_label = 'CC-Zero信息'
        self.draw_label(screen, widget_background, title_label, 300, 16, 10)
        info_label = f'MCTS搜索次数：{self.config.play.simulation_num_per_move}'
        self.draw_label(screen, widget_background, info_label, 335, 14, 10)
        eval_label = f"当前局势评估: {self.nn_value:.3f}"
        self.draw_label(screen, widget_background, eval_label, 360, 14, 10)
        label = f"MCTS搜索结果:"
        self.draw_label(screen, widget_background, label, 395, 14, 10)
        label = f"着法 访问计数 动作价值 先验概率"
        self.draw_label(screen, widget_background, label, 415, 12, 10)
        i = 0
        tmp = copy.deepcopy(self.mcts_moves)
        for mov, action_state in tmp.items():
            label = f"{mov}"
            self.draw_label(screen, widget_background, label, 435 + i * 20, 12, 10)
            label = f"{action_state[0]}"
            self.draw_label(screen, widget_background, label, 435 + i * 20, 12, 70)
            label = f"{action_state[1]:.2f}"
            self.draw_label(screen, widget_background, label, 435 + i * 20, 12, 100)
            label = f"{action_state[2]:.3f}"
            self.draw_label(screen, widget_background, label, 435 + i * 20, 12, 150)
            i += 1

    def draw_label(self, screen, widget_background, text, y, font_size, x=None):
        """
        工具函数：绘制单行label
        """
        font_file = self.config.resource.font_path
        font = pygame.font.Font(font_file, font_size)
        label = font.render(text, True, (0, 0, 0), (255, 255, 255))
        t_rect = label.get_rect()
        t_rect.y = y
        if x != None:
            t_rect.x = x
        else:
            t_rect.centerx = (self.screen_width - self.width) / 2
        widget_background.blit(label, t_rect)
        screen.blit(widget_background, (self.width, 0))


class Chessman_Sprite(pygame.sprite.Sprite):
    """
    pygame棋子精灵类
    """
    is_selected = False
    images = []
    is_transparent = False

    def __init__(self, images, chessman, w=80, h=80):
        pygame.sprite.Sprite.__init__(self)
        self.chessman = chessman
        self.images = [pygame.transform.scale(image, (w, h)) for image in images]
        self.image = self.images[0]
        self.rect = Rect(chessman.col_num * w, (9 - chessman.row_num) * h, w, h)

    def move(self, col_num, row_num, w=80, h=80):
        # 检查并执行棋子移动
        old_col_num = self.chessman.col_num
        old_row_num = self.chessman.row_num
        is_correct_position = self.chessman.move(col_num, row_num)
        if is_correct_position:
            self.rect = Rect(old_col_num * w, (9 - old_row_num) * h, w, h)
            self.rect.move_ip((col_num - old_col_num)
                              * w, (old_row_num - row_num) * h)
            self.chessman.chessboard.clear_chessmans_moving_list()
            self.chessman.chessboard.calc_chessmans_moving_list()
            return True
        return False

    def update(self):
        if self.is_selected:
            self.image = self.images[1]
        else:
            self.image = self.images[0]


def load_image(file, sub_dir=None):
    '''加载图片并返回Surface对象'''
    if sub_dir:
        file = os.path.join(main_dir, 'images', sub_dir, file)
    else:
        file = os.path.join(main_dir, 'images', file)
    try:
        surface = pygame.image.load(file)
    except pygame.error:
        raise SystemExit('Could not load image "%s" %s' %
                         (file, pygame.get_error()))
    return surface.convert()


def load_images(*files):
    """
    批量加载图片
    """
    global PIECE_STYLE
    imgs = []
    for file in files:
        imgs.append(load_image(file, PIECE_STYLE))
    return imgs


def creat_sprite_group(sprite_group, chessmans_hash, w, h):
    """
    将所有棋子对象生成pygame精灵，加入组
    """
    for chess in chessmans_hash.values():
        if chess.is_red:
            if isinstance(chess, Rook):
                images = load_images("RR.GIF", "RRS.GIF")
            elif isinstance(chess, Cannon):
                images = load_images("RC.GIF", "RCS.GIF")
            elif isinstance(chess, Knight):
                images = load_images("RN.GIF", "RNS.GIF")
            elif isinstance(chess, King):
                images = load_images("RK.GIF", "RKS.GIF")
            elif isinstance(chess, Elephant):
                images = load_images("RB.GIF", "RBS.GIF")
            elif isinstance(chess, Mandarin):
                images = load_images("RA.GIF", "RAS.GIF")
            else:
                images = load_images("RP.GIF", "RPS.GIF")
        else:
            if isinstance(chess, Rook):
                images = load_images("BR.GIF", "BRS.GIF")
            elif isinstance(chess, Cannon):
                images = load_images("BC.GIF", "BCS.GIF")
            elif isinstance(chess, Knight):
                images = load_images("BN.GIF", "BNS.GIF")
            elif isinstance(chess, King):
                images = load_images("BK.GIF", "BKS.GIF")
            elif isinstance(chess, Elephant):
                images = load_images("BB.GIF", "BBS.GIF")
            elif isinstance(chess, Mandarin):
                images = load_images("BA.GIF", "BAS.GIF")
            else:
                images = load_images("BP.GIF", "BPS.GIF")
        chessman_sprite = Chessman_Sprite(images, chess, w, h)
        sprite_group.add(chessman_sprite)


def select_sprite_from_group(sprite_group, col_num, row_num):
    """
    根据棋盘坐标选择对应精灵
    """
    for sprite in sprite_group:
        if sprite.chessman.col_num == col_num and sprite.chessman.row_num == row_num:
            return sprite
    return None


def translate_hit_area(screen_x, screen_y, w=80, h=80):
    """
    屏幕像素坐标转为棋盘格坐标
    """
    return screen_x // w, 9 - screen_y // h
