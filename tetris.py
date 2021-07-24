# -*- coding:utf-8 -*-
import tkinter as tk
from tkinter import font
import random

import tensorflow as tf
import cv2
import time
import argparse
import posenet
import csv
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--match_model', type=str, default='./output_csv/motion_model.csv')
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

def weightedDistanceMatching(poseVector1_x, poseVector1_y, vector1Confidences, vector1ConfidenceSum, poseVector2):
#   First summation
    summation1 = 1.0 / vector1ConfidenceSum
#   Second summation
    summation2 = 0
    for indent_num in range(len(poseVector1_x)):
        tempSum = vector1Confidences[indent_num] * ( abs(poseVector1_x[indent_num] -  poseVector2[indent_num]) + abs(poseVector1_y[indent_num] -  poseVector2[indent_num + len(poseVector1_x)]))
        summation2 = summation2 + tempSum
    return summation1 * summation2

#model_cfg, model_outputs = posenet.load_model(args.model, sess)
#output_stride = model_cfg['output_stride']


# 定数
BLOCK_SIZE = 25  # ブロックの縦横サイズpx
FIELD_WIDTH = 10  # フィールドの幅
FIELD_HEIGHT = 20  # フィールドの高さ

MOVE_LEFT = 0  # 左にブロックを移動することを示す定数
MOVE_RIGHT = 1  # 右にブロックを移動することを示す定数
MOVE_DOWN = 2  # 下にブロックを移動することを示す定数
MOVE_ROT = 3 #90°回転を示す定数
MOVE_ROT_INV = 4 #-90°回転を示す定数

score = 0
gameover = 0
life = 0

# ブロックを構成する正方形のクラス
class TetrisSquare():
    def __init__(self, x=0, y=0, color="gray"):
        '１つの正方形を作成'
        self.x = x
        self.y = y
        self.color = color

    def set_cord(self, x, y):
        '正方形の座標を設定'
        self.x = x
        self.y = y

    def get_cord(self):
        '正方形の座標を取得'
        return int(self.x), int(self.y)

    def set_color(self, color):
        '正方形の色を設定'
        self.color = color

    def get_color(self):
        '正方形の色を取得'
        return self.color

    def get_moved_cord(self, direction):
        '移動後の正方形の座標を取得'

        # 移動前の正方形の座標を取得
        x, y = self.get_cord()

        # 移動方向を考慮して移動後の座標を計算
        if direction == MOVE_LEFT:
            return x - 1, y
        elif direction == MOVE_RIGHT:
            return x + 1, y
        elif direction == MOVE_DOWN:
            return x, y + 1
        else:
            return x, y

    def get_rot_cord(self, direction, main_x, main_y):
        '回転後の正方形の座標を取得'

        # 移動前の正方形の座標を取得
        x, y = self.get_cord()

        # 移動方向を考慮して移動後の座標を計算
        if direction == MOVE_ROT:
            x = x - main_x
            y = y - main_y
            a = x
            x = -y
            y = a
            x = x + main_x
            y = y + main_y
            return x, y
        if direction == MOVE_ROT_INV:
            x = x - main_x
            y = y - main_y
            a = x
            x = y
            y = -a
            x = x + main_x
            y = y + main_y
            return x, y
        
        
# テトリス画面を描画するキャンバスクラス
class TetrisCanvas(tk.Canvas):
    def __init__(self, master, field):
        'テトリスを描画するキャンバスを作成'

        canvas_width = field.get_width() * BLOCK_SIZE 
        canvas_height = field.get_height() * BLOCK_SIZE

        # tk.Canvasクラスのinit
        super().__init__(master, width=canvas_width, height=canvas_height, bg="white")
    

        # キャンバスを画面上に設置
        self.place(x=25, y=25)

        # 10x20個の正方形を描画することでテトリス画面を作成
        for y in range(field.get_height()):
            for x in range(field.get_width()):
                square = field.get_square(x, y)
                x1 = x * BLOCK_SIZE
                x2 = (x + 1) * BLOCK_SIZE
                y1 = y * BLOCK_SIZE
                y2 = (y + 1) * BLOCK_SIZE
                self.create_rectangle(
                    x1, y1, x2, y2,
                    outline="white", width=1,
                    fill=square.get_color()
                )

        # 一つ前に描画したフィールドを設定
        self.before_field = field

    def update(self, field, block):
        'テトリス画面をアップデート'

        # 描画用のフィールド（フィールド＋ブロック）を作成
        new_field = TetrisField()
        for y in range(field.get_height()):
            for x in range(field.get_width()):
                square = field.get_square(x, y)
                color = square.get_color()

                new_square = new_field.get_square(x, y)
                new_square.set_color(color)

        # フィールドにブロックの正方形情報を合成
        if block is not None:
            block_squares = block.get_squares()
            for block_square in block_squares:
                # ブロックの正方形の座標と色を取得
                x, y = block_square.get_cord()
                color = block_square.get_color()

                # 取得した座標のフィールド上の正方形の色を更新
                new_field_square = new_field.get_square(x, y)
                new_field_square.set_color(color)

        # 描画用のフィールドを用いてキャンバスに描画
        for y in range(field.get_height()):
            for x in range(field.get_width()):

                # (x,y)座標のフィールドの色を取得
                new_square = new_field.get_square(x, y)
                new_color = new_square.get_color()

                # (x,y)座標が前回描画時から変化ない場合は描画しない
                before_square = self.before_field.get_square(x, y)
                before_color = before_square.get_color()
                if(new_color == before_color):
                    continue

                x1 = x * BLOCK_SIZE
                x2 = (x + 1) * BLOCK_SIZE
                y1 = y * BLOCK_SIZE
                y2 = (y + 1) * BLOCK_SIZE
                # フィールドの各位置の色で長方形描画
                self.create_rectangle(
                    x1, y1, x2, y2,
                    outline="white", width=1, fill=new_color
                )

        # 前回描画したフィールドの情報を更新
        self.before_field = new_field

# 積まれたブロックの情報を管理するフィールドクラス
class TetrisField():
    # score = 100
    def __init__(self):
        self.width = FIELD_WIDTH
        self.height = FIELD_HEIGHT

        # フィールドを初期化
        self.squares = []
        for y in range(self.height):
            for x in range(self.width):
                # フィールドを正方形インスタンスのリストとして管理
                self.squares.append(TetrisSquare(x, y, "gray"))

    def get_width(self):
        'フィールドの正方形の数（横方向）を取得'

        return self.width

    def get_height(self):
        'フィールドの正方形の数（縦方向）を取得'

        return self.height

    def get_squares(self):
        'フィールドを構成する正方形のリストを取得'

        return self.squares

    def get_square(self, x, y):
        '指定した座標の正方形を取得'

        return self.squares[y * self.width + x]

    def judge_game_over(self, block):
        # global gameover
        'ゲームオーバーかどうかを判断'

        # フィールド上で既に埋まっている座標の集合作成
        no_empty_cord = set(square.get_cord() for square
                            in self.get_squares() if square.get_color() != "gray")

        # ブロックがある座標の集合作成
        block_cord = set(square.get_cord() for square
                         in block.get_squares())

        # ブロックの座標の集合と
        # フィールドの既に埋まっている座標の集合の積集合を作成
        collision_set = no_empty_cord & block_cord

        # 積集合が空であればゲームオーバーではない
        if len(collision_set) == 0:
            ret = False
        else:
            ret = True

        return ret

    def judge_can_move(self, block, direction):
        '指定した方向にブロックを移動できるかを判断'

        # フィールド上で既に埋まっている座標の集合作成
        no_empty_cord = set(square.get_cord() for square
                            in self.get_squares() if square.get_color() != "gray")

        # 移動後のブロックがある座標の集合作成
        move_block_cord = set(square.get_moved_cord(direction) for square
                              in block.get_squares())

        # フィールドからはみ出すかどうかを判断
        for x, y in move_block_cord:

            # はみ出す場合は移動できない
            if x < 0 or x >= self.width or \
                    y < 0 or y >= self.height:
                return False

        # 移動後のブロックの座標の集合と
        # フィールドの既に埋まっている座標の集合の積集合を作成
        collision_set = no_empty_cord & move_block_cord

        # 積集合が空なら移動可能
        if len(collision_set) == 0:
            ret = True
        else:
            ret = False

        return ret
    
    

    def fix_block(self, block):
        'ブロックを固定してフィールドに追加'

        for square in block.get_squares():
            # ブロックに含まれる正方形の座標と色を取得
            x, y = square.get_cord()
            color = square.get_color()

            # その座標と色をフィールドに反映
            field_square = self.get_square(x, y)
            field_square.set_color(color)

    def delete_line(self):
        global score
        global var
        global life
        '行の削除を行う'
        # 全行に対して削除可能かどうかを調べていく
        for y in range(self.height):
            for x in range(self.width):
                # 行内に１つでも空があると消せない
                square = self.get_square(x, y)
                if(square.get_color() == "gray"):
                    # 次の行へ
                    break
            else:
                #SCORE = 10
                # break されなかった場合はその行は空きがない
                # この行を削除し、この行の上側にある行を１行下に移動
                for down_y in range(y, 0, -1):
                    for x in range(self.width):
                        src_square = self.get_square(x, down_y - 1)
                        dst_square = self.get_square(x, down_y)
                        dst_square.set_color(src_square.get_color())
                score += 1000
                if score % 3000 == 0:
                    life = 3
                    var_life.set("★★★")
                var.set("SCORE :  " + str(score))
                
                # 一番上の行は必ず全て空きになる
                for x in range(self.width):
                    square = self.get_square(x, 0)
                    square.set_color("gray")

# テトリスのブロックのクラス
class TetrisBlock():
    def __init__(self, chg, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9, vec10, vec11):
        'テトリスのブロックを作成'
        
        chg_vec = []
        chg_vec.append(vec1)
        chg_vec.append(vec2)
        chg_vec.append(vec3)
        chg_vec.append(vec4)
        chg_vec.append(vec5)
        chg_vec.append(vec6)
        chg_vec.append(vec7)
        chg_vec.append(vec8)
        chg_vec.append(vec9)
        chg_vec.append(vec10)
        chg_vec.append(vec11)
        
        chg_vec_x = []
        for num in chg_vec:
            if num == 2 or num == 3 or num == 4:
                chg_vec_x.append(1)
            elif num == 1 or num == 5:
                chg_vec_x.append(0)
            elif num == 6 or num == 7 or num == 8:
                chg_vec_x.append(-1)
        chg_vec_y = []
        for num in chg_vec:
            if num == 8 or num == 1 or num == 2:
                chg_vec_y.append(-1)
            elif num == 7 or num == 3:
                chg_vec_y.append(0)
            elif num == 4 or num == 5 or num == 6:
                chg_vec_y.append(1)
        
        # ブロックを構成する正方形のリスト
        self.squares = []

        if chg == 'I':
            block_type = 1
        elif chg == 'O':
            block_type = 2
        elif chg == 'L':
            block_type = 3
        elif chg == 'J':
            block_type = 4
        elif chg == 'Z':
            block_type = 5
        elif chg == 'S':
            block_type = 6
        elif chg == 'T':
            block_type = 7
        elif chg == 0:
            # ブロックの形をランダムに決定
            block_type = random.randint(1, 7)
        elif chg == 'N':
            block_type = 8
        
        
        # ブロックの形に応じて４つの正方形の座標と色を決定
        if block_type == 1:
            color = "red"
            cords = [
                [FIELD_WIDTH / 2, 0],
                [FIELD_WIDTH / 2, 1],
                [FIELD_WIDTH / 2, 2],
                [FIELD_WIDTH / 2, 3],
            ]
        elif block_type == 2:
            color = "blue"
            cords = [
                [FIELD_WIDTH / 2, 0],
                [FIELD_WIDTH / 2, 1],
                [FIELD_WIDTH / 2 - 1, 0],
                [FIELD_WIDTH / 2 - 1, 1],
            ]
        elif block_type == 3:
            color = "green"
            cords = [
                [FIELD_WIDTH / 2 - 1, 0],
                [FIELD_WIDTH / 2, 0],
                [FIELD_WIDTH / 2, 1],
                [FIELD_WIDTH / 2, 2],
            ]
        elif block_type == 4:
            color = "orange"
            cords = [
                [FIELD_WIDTH / 2, 0],
                [FIELD_WIDTH / 2 - 1, 0],
                [FIELD_WIDTH / 2 - 1, 1],
                [FIELD_WIDTH / 2 - 1, 2],
            ]
        elif block_type == 5:
            color = "yellow"
            cords = [
                [FIELD_WIDTH / 2, 0],
                [FIELD_WIDTH / 2, 1],
                [FIELD_WIDTH / 2 - 1, 1],
                [FIELD_WIDTH / 2 - 1, 2],
            ]
        elif block_type == 6:
            color = "brown"
            cords = [
                [FIELD_WIDTH / 2 - 1, 0],
                [FIELD_WIDTH / 2 - 1, 1],
                [FIELD_WIDTH / 2, 1],
                [FIELD_WIDTH / 2, 2],
            ]
        elif block_type == 7:
            color = "purple"
            cords = [
                [FIELD_WIDTH / 2, 0],
                [FIELD_WIDTH / 2, 1],
                [FIELD_WIDTH / 2 - 1, 1],
                [FIELD_WIDTH / 2, 2],
            ]
            
        elif block_type == 8:
            color = "black"
            cords = [
                [FIELD_WIDTH / 2, 1],
                [FIELD_WIDTH / 2 + chg_vec_x[0], 1 + chg_vec_y[0]],
                [FIELD_WIDTH / 2 + chg_vec_x[1], 1 + chg_vec_y[1]],
                [FIELD_WIDTH / 2 + chg_vec_x[2], 1 + chg_vec_y[2]],
                [FIELD_WIDTH / 2 + chg_vec_x[1] + chg_vec_x[3], 1 + chg_vec_y[1] + chg_vec_y[3]],
                [FIELD_WIDTH / 2 + chg_vec_x[2] + chg_vec_x[4], 1 + chg_vec_y[2] + chg_vec_y[4]],
                [FIELD_WIDTH / 2 + chg_vec_x[5], 1 + chg_vec_y[5]],
                [FIELD_WIDTH / 2 + chg_vec_x[5] + chg_vec_x[10], 1 + chg_vec_y[5] + chg_vec_y[10]],
                [FIELD_WIDTH / 2 + chg_vec_x[5] + chg_vec_x[10] + chg_vec_x[6], 1 + chg_vec_y[5] + chg_vec_y[10] + chg_vec_y[6]],
                [FIELD_WIDTH / 2 + chg_vec_x[5] + chg_vec_x[10] + chg_vec_x[7], 1 + chg_vec_y[5] + chg_vec_y[10] + chg_vec_y[7]],
                [FIELD_WIDTH / 2 + chg_vec_x[5] + chg_vec_x[10] + chg_vec_x[6] + chg_vec_x[8], 1 + chg_vec_y[5] + chg_vec_y[10] + chg_vec_y[6] + chg_vec_y[8]],
                [FIELD_WIDTH / 2 + chg_vec_x[5] + chg_vec_x[10] + chg_vec_x[7] + chg_vec_x[9], 1 + chg_vec_y[5] + chg_vec_y[10] + chg_vec_y[7] + chg_vec_y[9]],
            ]

        # 決定した色と座標の正方形を作成してリストに追加
        for cord in cords:
            self.squares.append(TetrisSquare(cord[0], cord[1], color))

    def get_squares(self):
        'ブロックを構成する正方形を取得'

        # return [square for square in self.squares]
        return self.squares

    def move(self, direction):
        'ブロックを移動'

        # ブロックを構成する正方形を移動
        for square in self.squares:
            x, y = square.get_moved_cord(direction)
            square.set_cord(x, y)

    def rot(self, direction):
        'ブロックの回転'
        a = [[0,0]]

        for square in self.squares:
            x, y = square.get_cord()
            b = [x,y]
            a.append(b)

        a = np.delete(a, 0, 0)

        main_x = a[2][0]
        main_y = a[2][1]

        for square in self.squares:
            x, y = square.get_rot_cord(direction,main_x,main_y)
            square.set_cord(x, y)

            
# テトリスゲームを制御するクラス
class TetrisGame():

    def __init__(self, master):
        'テトリスのインスタンス作成'

        # ブロック管理リストを初期化
        self.field = TetrisField()

        # 落下ブロックをセット
        self.block = None

        # テトリス画面をセット
        self.canvas = TetrisCanvas(master, self.field)

        # テトリス画面アップデート
        self.canvas.update(self.field, self.block)

    def start(self, func):
        'テトリスを開始'

        # 終了時に呼び出す関数をセット
        self.end_func = func

        # ブロック管理リストを初期化
        self.field = TetrisField()

        # 落下ブロックを新規追加
        self.new_block()
        
    def stop(self, new_mino, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9, vec10, vec11 ):
        'テトリスを中断'

        # 終了時に呼び出す関数をセット
        # self.end_func = func

        # ブロック管理リストを初期化
        # self.field = TetrisField()

        # 落下ブロックを新規追加
        self.chg_block(new_mino, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9, vec10, vec11 )

    def new_block(self, chg=0, vec1=0, vec2=0, vec3=0, vec4=0, vec5=0, vec6=0, vec7=0, vec8=0, vec9=0, vec10=0, vec11=0):
        'ブロックを新規追加'
        global gameover

        # 落下中のブロックインスタンスを作成
        self.block = TetrisBlock(chg, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9, vec10, vec11)

        if self.field.judge_game_over(self.block):
            self.end_func()
            gameover = 1
            var_gameover.set("GAMEOVER")
            

        # テトリス画面をアップデート
        self.canvas.update(self.field, self.block)
        
    def chg_block(self, new_mino, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9, vec10, vec11 ):
        # print(new_mino)
        self.block = TetrisBlock(new_mino, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9, vec10, vec11 )
        self.canvas.update(self.field, self.block)

    def move_block(self, direction):
        'ブロックを移動'

        # 移動できる場合だけ移動する
        if self.field.judge_can_move(self.block, direction):

            # ブロックを移動
            self.block.move(direction)

            # 画面をアップデート
            self.canvas.update(self.field, self.block)

        else:
            # ブロックが下方向に移動できなかった場合
            if direction == MOVE_DOWN:
                # ブロックを固定する
                self.field.fix_block(self.block)
                self.field.delete_line()
                self.new_block()
                
    def rot_block(self,direction):
        'ブロックを回転'
        self.block.rot(direction)

        # 画面をアップデート
        self.canvas.update(self.field, self.block)

# イベントを受け付けてそのイベントに応じてテトリスを制御するクラス
class EventHandller():
    def __init__(self, master, game):
        self.master = master

        # 制御するゲーム
        self.game = game

        # イベントを定期的に発行するタイマー
        self.timer = None

        # ゲームスタートボタンを設置
        start_button = tk.Button(master, text='START', command=self.start_event)
        start_button.place(x=25 + BLOCK_SIZE * FIELD_WIDTH + 25, y=30)

        stop_button = tk.Button(master, text='STOP', command=self.stop_event)
        stop_button.place(x=25 + BLOCK_SIZE * FIELD_WIDTH + 25, y=60)

        resume_button = tk.Button(master, text='RESUME', command=self.resume_event)
        resume_button.place(x=25 + BLOCK_SIZE * FIELD_WIDTH + 25, y=90)
        """
        stop_demo_button = tk.Button(master, text='STOP(demo)', command=self.stop_demo_event)
        stop_demo_button.place(x=25 + BLOCK_SIZE * FIELD_WIDTH + 25, y=120)
        """
        ##score_text = tk.Text()
        
    def start_event(self):
        'ゲームスタートボタンを押された時の処理'
        global gameover
        global score
        global life
        # テトリス開始
        self.game.start(self.end_event)
        self.running = True

        # タイマーセット
        self.timer_start()

        # キー操作入力受付開始
        self.master.bind("<Left>", self.left_key_event)
        self.master.bind("<Right>", self.right_key_event)
        self.master.bind("<space>", self.space_key_event)
        self.master.bind("<Down>", self.down_key_event)
        self.master.bind("<Up>", self.up_key_event)
        score = 0
        var.set("SCORE :  " + str(score))
        gameover = 0
        var_gameover.set("                      ")
        life = 3
        var_life.set("★★★")
        
        
        
    def resume_event(self):
        'ゲーム再開ボタンを押された時の処理'

        # テトリス開始
        self.running = True

        # タイマーセット
        self.timer_start()

    def end_event(self):
        'ゲーム終了時の処理'
        self.running = False

        # イベント受付を停止
        self.timer_end()
        self.master.unbind("<Left>")
        self.master.unbind("<Right>")
        self.master.unbind("<space>")
        self.master.unbind("<Down>")
        self.master.unbind("<Up>")

    def stop_event(self):
        'ゲーム終了時の処理'
        global life
        if life >= 1:
            
            self.running = False

            # イベント受付を停止
            self.timer_end()

            with tf.Session() as sess:
                with open(args.match_model) as f:
                    reader = csv.reader(f)
                    motion_model = [row for row in reader]
                for i in range(len(motion_model)):
                    motion_model[i][1:] = list(map(lambda x:float(x), motion_model[i][1:]))

                model_cfg, model_outputs = posenet.load_model(args.model, sess)
                output_stride = model_cfg['output_stride']

                if args.file is not None:
                    cap = cv2.VideoCapture(args.file)
                else:
                    cap = cv2.VideoCapture(args.cam_id)
                cap.set(3, args.cam_width)
                cap.set(4, args.cam_height)

                start = time.time()
                frame_count = 0
                vec1 = 0
                vec2 = 0
                vec3 = 0
                vec4 = 0
                vec5 = 0
                vec6 = 0
                vec7 = 0
                vec8 = 0
                vec9 = 0
                vec10 = 0
                """
                life -= 1
                if life == 2:
                    var_life.set("★★☆")
                if life == 1:
                    var_life.set("★☆☆")
                if life == 0:
                    var_life.set("☆☆☆")
                """


                while True:
                    input_image, display_image, output_scale = posenet.read_cap(
                        cap, scale_factor=args.scale_factor, output_stride=output_stride)

                    heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                        model_outputs,
                        feed_dict={'image:0': input_image}
                    )

                    pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                        heatmaps_result.squeeze(axis=0),
                        offsets_result.squeeze(axis=0),
                        displacement_fwd_result.squeeze(axis=0),
                        displacement_bwd_result.squeeze(axis=0),
                        output_stride=output_stride,
                        max_pose_detections=10,
                        min_pose_score=0.15)

                    keypoint_coords *= output_scale

                    if pose_scores[0] > 0.3:

                        # clip
                        keypoint_coords[0,:,0] = keypoint_coords[0,:,0] - min(keypoint_coords[0,:,0])
                        keypoint_coords[0,:,1] = keypoint_coords[0,:,1] - min(keypoint_coords[0,:,1])

                        key = max(keypoint_coords[0,:,1])

                        keypoint_coords[0,:,0] = keypoint_coords[0,:,0]/key
                        keypoint_coords[0,:,1] = keypoint_coords[0,:,1]/key

                        # normalize
                        x_l2_norm = np.linalg.norm(keypoint_coords[0,:,0],ord=2)
                        #pose_coords_x = (keypoint_coords[0,:,0] / x_l2_norm).tolist()
                        pose_coords_x = (keypoint_coords[0,:,0]).tolist()
                        y_l2_norm = np.linalg.norm(keypoint_coords[0,:,1],ord=2)
                        #pose_coords_y = (keypoint_coords[0,:,1] / y_l2_norm).tolist()
                        pose_coords_y = (keypoint_coords[0,:,1]).tolist()
                        distance_min = 1000000
                        min_num = -1
                        distance_I = 0
                        distance_L = 0
                        distance_J = 0
                        distance_Z = 0
                        distance_S = 0
                        distance_T = 0
                        distance_pose = [0,0,0,0,0,0]
                        
                        for teach_num in range(len(motion_model)):
                            ##print(motion_model[teach_num],"a")
                            distance = weightedDistanceMatching(pose_coords_x, pose_coords_y, keypoint_scores[0,:], pose_scores[0], motion_model[teach_num][1:35])
                            if motion_model[teach_num][0][1:2] == "I":
                                distance_I += distance
                            if motion_model[teach_num][0][1:2] == "J":
                                distance_J += distance
                            if motion_model[teach_num][0][1:2] == "S":
                                distance_S += distance
                            if motion_model[teach_num][0][1:2] == "Z":
                                distance_Z += distance
                            if motion_model[teach_num][0][1:2] == "L":
                                distance_L += distance
                            if motion_model[teach_num][0][1:2] == "T":
                                distance_T += distance
                        """        
                        print("I",distance_I)
                        print("J",distance_J)
                        print("S",distance_S)
                        print("Z",distance_Z)
                        print("L",distance_L)
                        print("T",distance_T)
                        """
                        distance_pose = [distance_I,distance_J,distance_S,distance_Z,distance_L,distance_T]
                        print(distance_pose)
                        if min(distance_pose) == distance_I:
                            distance = distance_I
                            pose = "I"
                        if min(distance_pose) == distance_J:
                            distance = distance_J
                            pose = "J"
                        if min(distance_pose) == distance_S:
                            distance = distance_S
                            pose = "S"
                        if min(distance_pose) == distance_Z:
                            distance = distance_Z
                            pose = "Z"
                        if min(distance_pose) == distance_L:
                            distance = distance_L
                            pose = "L"
                        if min(distance_pose) == distance_T:
                            distance = distance_T
                            pose = "T"
                            
    #                        print(motion_model[min_num][0],distance)
                        if distance < 40:
                            cv2.putText(display_image, "success!! pose: "+ pose, (50,100), cv2.FONT_HERSHEY_PLAIN, 8.0, (0, 100, 255), thickness=8)
                            #cv2.putText(display_image, motion_model[min_num][0][1:2], (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), thickness=2)
                        else:
                            cv2.putText(display_image, "Miss! Penalty!", (200,100), cv2.FONT_HERSHEY_PLAIN, 8.0, (0, 100, 255), thickness=8)
        # int(keypoint_coords[0,:,0][0]),int(keypoint_coords[0,:,0][1])=(100,100)

       #                 print(motion_model[min_num][0][1:2])

                    overlay_image = posenet.draw_skel_and_kp(
                        display_image, pose_scores, keypoint_scores, keypoint_coords,
                        min_pose_score=0.15, min_part_score=0.1)

                    cv2.imshow('posenet', overlay_image)
                    # TODO this isn't particularly fast, use GL for drawing and display someday...

                    frame_count += 1
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    limit_time = int(time.time()-start)
        #            print(limit_time)

                    if limit_time >= 10:
                        # print("time>10!!")
                        #print(motion_model[min_num][0][1:2],distance)
                        ###print(keypoint_coords[0,:,:])
                        ####未知のブロック生成(for文にすれば20行で終わります)
                        shoulder = (keypoint_coords[0,5,:]+keypoint_coords[0,6,:])/2
                        hip = (keypoint_coords[0,11,:]+keypoint_coords[0,12,:])/2
                        heso = (hip+shoulder)/2
                        vec_1 = keypoint_coords[0,0,:]-shoulder
                        vec_2 = keypoint_coords[0,7,:]-shoulder
                        vec_3 = keypoint_coords[0,8,:]-shoulder
                        vec_4 = keypoint_coords[0,9,:]-keypoint_coords[0,7,:]
                        vec_5 = keypoint_coords[0,10,:]-keypoint_coords[0,8,:]
                        vec_6 = heso-shoulder
                        vec_11 = hip-heso
                        vec_7 = keypoint_coords[0,13,:]-hip
                        vec_8 = keypoint_coords[0,14,:]-hip
                        vec_9 = keypoint_coords[0,15,:]-keypoint_coords[0,13,:]
                        vec_10 = keypoint_coords[0,16,:]-keypoint_coords[0,14,:]
                        slope_1 = -vec_1[0]/vec_1[1]
                        slope_2 = -vec_2[0]/vec_2[1]
                        slope_3 = -vec_3[0]/vec_3[1]
                        slope_4 = -vec_4[0]/vec_4[1]
                        slope_5 = -vec_5[0]/vec_5[1]
                        slope_6 = -vec_6[0]/vec_6[1]
                        slope_7 = -vec_7[0]/vec_7[1]
                        slope_8 = -vec_8[0]/vec_8[1]
                        slope_9 = -vec_9[0]/vec_9[1]
                        slope_10 = -vec_10[0]/vec_10[1]
                        slope_11 = -vec_11[0]/vec_11[1]



                        if slope_1 < -2.4:
                            if vec_1[0] >= 0:
                                vec1 = 5
                            elif vec_1[0] < 0:
                                vec1 = 1
                        elif slope_1 < -0.41:
                            if vec_1[0] >= 0:
                                vec1 = 4
                            elif vec_1[0] < 0:
                                vec1 = 8
                        elif slope_1 < 0.41:
                            if vec_1[1] >= 0:
                                vec1 = 3
                            elif vec_1[1] < 0:
                                vec1 = 7
                        elif slope_1 < 2.4:
                            if vec_1[1] >= 0:
                                vec1 = 2
                            elif vec_1[1] < 0:
                                vec1 = 6
                        else:
                            if vec_1[0] >= 0:
                                vec1 = 5
                            elif vec_1[0] < 0:
                                vec1 = 1

                        if slope_2 < -2.4:
                            if vec_2[0] >= 0:
                                vec2 = 5
                            elif vec_2[0] < 0:
                                vec2 = 1
                        elif slope_2 < -0.41:
                            if vec_2[0] >= 0:
                                vec2 = 4
                            elif vec_2[0] < 0:
                                vec2 = 8
                        elif slope_2 < 0.41:
                            if vec_2[1] >= 0:
                                vec2 = 3
                            elif vec_2[1] < 0:
                                vec2 = 7
                        elif slope_2 < 2.4:
                            if vec_2[1] >= 0:
                                vec2 = 2
                            elif vec_2[1] < 0:
                                vec2 = 6
                        else:
                            if vec_2[0] >= 0:
                                vec2 = 5
                            elif vec_2[0] < 0:
                                vec2 = 1

                        if slope_3 < -2.4:
                            if vec_3[0] >= 0:
                                vec3 = 5
                            elif vec_3[0] < 0:
                                vec3 = 1
                        elif slope_3 < -0.41:
                            if vec_3[0] >= 0:
                                vec3 = 4
                            elif vec_3[0] < 0:
                                vec3 = 8
                        elif slope_3 < 0.41:
                            if vec_3[1] >= 0:
                                vec3 = 3
                            elif vec_3[1] < 0:
                                vec3 = 7
                        elif slope_3 < 2.4:
                            if vec_3[1] >= 0:
                                vec3 = 2
                            elif vec_3[1] < 0:
                                vec3 = 6
                        else:
                            if vec_3[0] >= 0:
                                vec3 = 5
                            elif vec_3[0] < 0:
                                vec3 = 1

                        if slope_4 < -2.4:
                            if vec_4[0] >= 0:
                                vec4 = 5
                            elif vec_4[0] < 0:
                                vec4 = 1
                        elif slope_4 < -0.41:
                            if vec_4[0] >= 0:
                                vec4 = 4
                            elif vec_4[0] < 0:
                                vec4 = 8
                        elif slope_4 < 0.41:
                            if vec_4[1] >= 0:
                                vec4 = 3
                            elif vec_4[1] < 0:
                                vec4 = 7
                        elif slope_4 < 2.4:
                            if vec_4[1] >= 0:
                                vec4 = 2
                            elif vec_4[1] < 0:
                                vec4 = 6
                        else:
                            if vec_4[0] >= 0:
                                vec4 = 5
                            elif vec_4[0] < 0:
                                vec4 = 1

                        if slope_5 < -2.4:
                            if vec_5[0] >= 0:
                                vec5 = 5
                            elif vec_5[0] < 0:
                                vec5 = 1
                        elif slope_5 < -0.41:
                            if vec_5[0] >= 0:
                                vec5 = 4
                            elif vec_5[0] < 0:
                                vec5 = 8
                        elif slope_5 < 0.41:
                            if vec_5[1] >= 0:
                                vec5 = 3
                            elif vec_5[1] < 0:
                                vec5 = 7
                        elif slope_5 < 2.4:
                            if vec_5[1] >= 0:
                                vec5 = 2
                            elif vec_5[1] < 0:
                                vec5 = 6
                        else:
                            if vec_5[0] >= 0:
                                vec5 = 5
                            elif vec_5[0] < 0:
                                vec5 = 1

                        if slope_6 < -2.4:
                            if vec_6[0] >= 0:
                                vec6 = 5
                            elif vec_6[0] < 0:
                                vec6 = 1
                        elif slope_6 < -0.41:
                            if vec_6[0] >= 0:
                                vec6 = 4
                            elif vec_6[0] < 0:
                                vec6 = 8
                        elif slope_6 < 0.41:
                            if vec_6[1] >= 0:
                                vec6 = 3
                            elif vec_6[1] < 0:
                                vec6 = 7
                        elif slope_6 < 2.4:
                            if vec_6[1] >= 0:
                                vec6 = 2
                            elif vec_6[1] < 0:
                                vec6 = 6
                        else:
                            if vec_6[0] >= 0:
                                vec6 = 5
                            elif vec_6[0] < 0:
                                vec6 = 1

                        if slope_7 < -2.4:
                            if vec_7[0] >= 0:
                                vec7 = 5
                            elif vec_7[0] < 0:
                                vec7 = 1
                        elif slope_7 < -0.41:
                            if vec_7[0] >= 0:
                                vec7 = 4
                            elif vec_7[0] < 0:
                                vec7 = 8
                        elif slope_7 < 0.41:
                            if vec_7[1] >= 0:
                                vec7 = 3
                            elif vec_7[1] < 0:
                                vec7 = 7
                        elif slope_7 < 2.4:
                            if vec_7[1] >= 0:
                                vec7 = 2
                            elif vec_7[1] < 0:
                                vec7 = 6
                        else:
                            if vec_7[0] >= 0:
                                vec7 = 5
                            elif vec_7[0] < 0:
                                vec7 = 1

                        if slope_8 < -2.4:
                            if vec_8[0] >= 0:
                                vec8 = 5
                            elif vec_8[0] < 0:
                                vec8 = 1
                        elif slope_8 < -0.41:
                            if vec_8[0] >= 0:
                                vec8 = 4
                            elif vec_8[0] < 0:
                                vec8 = 8
                        elif slope_8 < 0.41:
                            if vec_8[1] >= 0:
                                vec8 = 3
                            elif vec_8[1] < 0:
                                vec8 = 7
                        elif slope_8 < 2.4:
                            if vec_8[1] >= 0:
                                vec8 = 2
                            elif vec_8[1] < 0:
                                vec8 = 6
                        else:
                            if vec_8[0] >= 0:
                                vec8 = 5
                            elif vec_8[0] < 0:
                                vec8 = 1

                        if slope_9 < -2.4:
                            if vec_9[0] >= 0:
                                vec9 = 5
                            elif vec_9[0] < 0:
                                vec9 = 1
                        elif slope_9 < -0.41:
                            if vec_9[0] >= 0:
                                vec9 = 4
                            elif vec_9[0] < 0:
                                vec9 = 8
                        elif slope_9 < 0.41:
                            if vec_9[1] >= 0:
                                vec9 = 3
                            elif vec_9[1] < 0:
                                vec9 = 7
                        elif slope_9 < 2.4:
                            if vec_9[1] >= 0:
                                vec9 = 2
                            elif vec_9[1] < 0:
                                vec9 = 6
                        else:
                            if vec_9[0] >= 0:
                                vec9 = 5
                            elif vec_9[0] < 0:
                                vec9 = 1

                        if slope_10 < -2.4:
                            if vec_10[0] >= 0:
                                vec10 = 5
                            elif vec_10[0] < 0:
                                vec10 = 1
                        elif slope_10 < -0.41:
                            if vec_10[0] >= 0:
                                vec10 = 4
                            elif vec_10[0] < 0:
                                vec10 = 8
                        elif slope_10 < 0.41:
                            if vec_10[1] >= 0:
                                vec10 = 3
                            elif vec_10[1] < 0:
                                vec10 = 7
                        elif slope_10 < 2.4:
                            if vec_10[1] >= 0:
                                vec10 = 2
                            elif vec_10[1] < 0:
                                vec10 = 6
                        else:
                            if vec_10[0] >= 0:
                                vec10 = 5
                            elif vec_10[0] < 0:
                                vec10 = 1

                        if slope_11 < -2.4:
                            if vec_11[0] >= 0:
                                vec11 = 5
                            elif vec_11[0] < 0:
                                vec11 = 1
                        elif slope_11 < -0.41:
                            if vec_11[0] >= 0:
                                vec11 = 4
                            elif vec_11[0] < 0:
                                vec11 = 8
                        elif slope_11 < 0.41:
                            if vec_11[1] >= 0:
                                vec11 = 3
                            elif vec_11[1] < 0:
                                vec11 = 7
                        elif slope_11 < 2.4:
                            if vec_11[1] >= 0:
                                vec11 = 2
                            elif vec_11[1] < 0:
                                vec11 = 6
                        else:
                            if vec_11[0] >= 0:
                                vec11 = 5
                            elif vec_10[0] < 0:
                                vec11 = 1

                        life -= 1
                        if life == 2:
                            var_life.set("★★☆")
                        if life == 1:
                            var_life.set("★☆☆")
                        if life == 0:
                            var_life.set("☆☆☆")
                            
                        ####
                        if distance >= 40:
                            self.game.stop("N",vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9, vec10, vec11)
                        if distance < 40:
                            self.game.stop(pose,vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9, vec10, vec11 )                        
                        return

        
    def stop_demo_event(self):
        'ゲーム終了時の処理'
        global life
        if life >= 1:
            
            self.running = False

            # イベント受付を停止
            self.timer_end()

            with tf.Session() as sess:
                with open(args.match_model) as f:
                    reader = csv.reader(f)
                    motion_model = [row for row in reader]
                for i in range(len(motion_model)):
                    motion_model[i][1:] = list(map(lambda x:float(x), motion_model[i][1:]))

                model_cfg, model_outputs = posenet.load_model(args.model, sess)
                output_stride = model_cfg['output_stride']

                if args.file is not None:
                    cap = cv2.VideoCapture(args.file)
                else:
                    cap = cv2.VideoCapture(args.cam_id)
                cap.set(3, args.cam_width)
                cap.set(4, args.cam_height)

                start = time.time()
                frame_count = 0
                vec1 = 0
                vec2 = 0
                vec3 = 0
                vec4 = 0
                vec5 = 0
                vec6 = 0
                vec7 = 0
                vec8 = 0
                vec9 = 0
                vec10 = 0
                
                life -= 1
                if life == 2:
                    var_life.set("★★☆")
                if life == 1:
                    var_life.set("★☆☆")
                if life == 0:
                    var_life.set("☆☆☆")



                while True:
                    input_image, display_image, output_scale = posenet.read_cap(
                        cap, scale_factor=args.scale_factor, output_stride=output_stride)

                    heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                        model_outputs,
                        feed_dict={'image:0': input_image}
                    )

                    pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                        heatmaps_result.squeeze(axis=0),
                        offsets_result.squeeze(axis=0),
                        displacement_fwd_result.squeeze(axis=0),
                        displacement_bwd_result.squeeze(axis=0),
                        output_stride=output_stride,
                        max_pose_detections=10,
                        min_pose_score=0.15)

                    keypoint_coords *= output_scale

                    if pose_scores[0] > 0.3:
                        # clip
                        pose_coords_x = keypoint_coords[0,:,0] - min(keypoint_coords[0,:,0])
                        pose_coords_y = keypoint_coords[0,:,1] - min(keypoint_coords[0,:,1])

                        # normalize
                        x_l2_norm = np.linalg.norm(keypoint_coords[0,:,0],ord=2)
                        pose_coords_x = (pose_coords_x / x_l2_norm).tolist()
                        y_l2_norm = np.linalg.norm(keypoint_coords[0,:,1],ord=2)
                        pose_coords_y = (pose_coords_y / y_l2_norm).tolist()

                        distance_min = 1000000
                        min_num = -1
                        distance_I = 0
                        distance_L = 0
                        distance_J = 0
                        distance_Z = 0
                        distance_S = 0
                        distance_T = 0
                        distance_pose = [0,0,0,0,0,0]
                        
                        """
        #                motion_model[min_num][0]="Miss"
                        for teach_num in range(len(motion_model)):
                            print(motion_model[teach_num][0][1:2],"a")
                            distance = weightedDistanceMatching(pose_coords_x, pose_coords_y, keypoint_scores[0,:], pose_scores[0], motion_model[teach_num][1:35])
                            # distance = cos_sim(pose_coords_x + pose_coords_y, motion_model[teach_num][1:35])
                            if distance < distance_min:
                                distance_min = distance
                                min_num = teach_num
                        """
                        for teach_num in range(len(motion_model)):
                            distance = weightedDistanceMatching(pose_coords_x, pose_coords_y, keypoint_scores[0,:], pose_scores[0], motion_model[teach_num][1:35])
                            if motion_model[teach_num][0][1:2] == "I":
                                distance_I += distance
                            if motion_model[teach_num][0][1:2] == "J":
                                distance_J += distance
                            if motion_model[teach_num][0][1:2] == "S":
                                distance_S += distance
                            if motion_model[teach_num][0][1:2] == "Z":
                                distance_Z += distance
                            if motion_model[teach_num][0][1:2] == "L":
                                distance_L += distance
                            if motion_model[teach_num][0][1:2] == "T":
                                distance_T += distance

                        distance_pose = [distance_I,distance_J,distance_S,distance_Z,distance_L,distance_T]
                        if min(distance_pose) == distance_I:
                            distance = distance_I
                            pose = "I"
                        if min(distance_pose) == distance_J:
                            distance = distance_J
                            pose = "J"
                        if min(distance_pose) == distance_S:
                            distance = distance_S
                            pose = "S"
                        if min(distance_pose) == distance_Z:
                            distance = distance_Z
                            pose = "Z"
                        if min(distance_pose) == distance_L:
                            distance = distance_L
                            pose = "L"
                        if min(distance_pose) == distance_T:
                            distance = distance_T
                            pose = "T"
                            
    #                        print(motion_model[min_num][0],distance)
    
                        if distance < 0:
                            cv2.putText(display_image, pose, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), thickness=2)
                            #cv2.putText(display_image, motion_model[min_num][0][1:2], (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), thickness=2)
                        else:
                            cv2.putText(display_image, "Miss", (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), thickness=2)
        # int(keypoint_coords[0,:,0][0]),int(keypoint_coords[0,:,0][1])=(100,100)

       #                 print(motion_model[min_num][0][1:2])

                    overlay_image = posenet.draw_skel_and_kp(
                        display_image, pose_scores, keypoint_scores, keypoint_coords,
                        min_pose_score=0.15, min_part_score=0.1)

                    cv2.imshow('posenet', overlay_image)
                    # TODO this isn't particularly fast, use GL for drawing and display someday...

                    frame_count += 1
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    limit_time = int(time.time()-start)
        #            print(limit_time)

                    if limit_time >= 10:
                        # print("time>10!!")
                        #print(motion_model[min_num][0][1:2],distance)
                        ###print(keypoint_coords[0,:,:])
                        ####未知のブロック生成(for文にすれば20行で終わります)
                        shoulder = (keypoint_coords[0,5,:]+keypoint_coords[0,6,:])/2
                        hip = (keypoint_coords[0,11,:]+keypoint_coords[0,12,:])/2
                        heso = (hip+shoulder)/2
                        vec_1 = keypoint_coords[0,0,:]-shoulder
                        vec_2 = keypoint_coords[0,7,:]-shoulder
                        vec_3 = keypoint_coords[0,8,:]-shoulder
                        vec_4 = keypoint_coords[0,9,:]-keypoint_coords[0,7,:]
                        vec_5 = keypoint_coords[0,10,:]-keypoint_coords[0,8,:]
                        vec_6 = heso-shoulder
                        vec_11 = hip-heso
                        vec_7 = keypoint_coords[0,13,:]-hip
                        vec_8 = keypoint_coords[0,14,:]-hip
                        vec_9 = keypoint_coords[0,15,:]-keypoint_coords[0,13,:]
                        vec_10 = keypoint_coords[0,16,:]-keypoint_coords[0,14,:]
                        slope_1 = -vec_1[0]/vec_1[1]
                        slope_2 = -vec_2[0]/vec_2[1]
                        slope_3 = -vec_3[0]/vec_3[1]
                        slope_4 = -vec_4[0]/vec_4[1]
                        slope_5 = -vec_5[0]/vec_5[1]
                        slope_6 = -vec_6[0]/vec_6[1]
                        slope_7 = -vec_7[0]/vec_7[1]
                        slope_8 = -vec_8[0]/vec_8[1]
                        slope_9 = -vec_9[0]/vec_9[1]
                        slope_10 = -vec_10[0]/vec_10[1]
                        slope_11 = -vec_11[0]/vec_11[1]

                        if slope_1 < -2.4:
                            if vec_1[0] >= 0:
                                vec1 = 5
                            elif vec_1[0] < 0:
                                vec1 = 1
                        elif slope_1 < -0.41:
                            if vec_1[0] >= 0:
                                vec1 = 4
                            elif vec_1[0] < 0:
                                vec1 = 8
                        elif slope_1 < 0.41:
                            if vec_1[1] >= 0:
                                vec1 = 3
                            elif vec_1[1] < 0:
                                vec1 = 7
                        elif slope_1 < 2.4:
                            if vec_1[1] >= 0:
                                vec1 = 2
                            elif vec_1[1] < 0:
                                vec1 = 6
                        else:
                            if vec_1[0] >= 0:
                                vec1 = 5
                            elif vec_1[0] < 0:
                                vec1 = 1

                        if slope_2 < -2.4:
                            if vec_2[0] >= 0:
                                vec2 = 5
                            elif vec_2[0] < 0:
                                vec2 = 1
                        elif slope_2 < -0.41:
                            if vec_2[0] >= 0:
                                vec2 = 4
                            elif vec_2[0] < 0:
                                vec2 = 8
                        elif slope_2 < 0.41:
                            if vec_2[1] >= 0:
                                vec2 = 3
                            elif vec_2[1] < 0:
                                vec2 = 7
                        elif slope_2 < 2.4:
                            if vec_2[1] >= 0:
                                vec2 = 2
                            elif vec_2[1] < 0:
                                vec2 = 6
                        else:
                            if vec_2[0] >= 0:
                                vec2 = 5
                            elif vec_2[0] < 0:
                                vec2 = 1

                        if slope_3 < -2.4:
                            if vec_3[0] >= 0:
                                vec3 = 5
                            elif vec_3[0] < 0:
                                vec3 = 1
                        elif slope_3 < -0.41:
                            if vec_3[0] >= 0:
                                vec3 = 4
                            elif vec_3[0] < 0:
                                vec3 = 8
                        elif slope_3 < 0.41:
                            if vec_3[1] >= 0:
                                vec3 = 3
                            elif vec_3[1] < 0:
                                vec3 = 7
                        elif slope_3 < 2.4:
                            if vec_3[1] >= 0:
                                vec3 = 2
                            elif vec_3[1] < 0:
                                vec3 = 6
                        else:
                            if vec_3[0] >= 0:
                                vec3 = 5
                            elif vec_3[0] < 0:
                                vec3 = 1

                        if slope_4 < -2.4:
                            if vec_4[0] >= 0:
                                vec4 = 5
                            elif vec_4[0] < 0:
                                vec4 = 1
                        elif slope_4 < -0.41:
                            if vec_4[0] >= 0:
                                vec4 = 4
                            elif vec_4[0] < 0:
                                vec4 = 8
                        elif slope_4 < 0.41:
                            if vec_4[1] >= 0:
                                vec4 = 3
                            elif vec_4[1] < 0:
                                vec4 = 7
                        elif slope_4 < 2.4:
                            if vec_4[1] >= 0:
                                vec4 = 2
                            elif vec_4[1] < 0:
                                vec4 = 6
                        else:
                            if vec_4[0] >= 0:
                                vec4 = 5
                            elif vec_4[0] < 0:
                                vec4 = 1

                        if slope_5 < -2.4:
                            if vec_5[0] >= 0:
                                vec5 = 5
                            elif vec_5[0] < 0:
                                vec5 = 1
                        elif slope_5 < -0.41:
                            if vec_5[0] >= 0:
                                vec5 = 4
                            elif vec_5[0] < 0:
                                vec5 = 8
                        elif slope_5 < 0.41:
                            if vec_5[1] >= 0:
                                vec5 = 3
                            elif vec_5[1] < 0:
                                vec5 = 7
                        elif slope_5 < 2.4:
                            if vec_5[1] >= 0:
                                vec5 = 2
                            elif vec_5[1] < 0:
                                vec5 = 6
                        else:
                            if vec_5[0] >= 0:
                                vec5 = 5
                            elif vec_5[0] < 0:
                                vec5 = 1

                        if slope_6 < -2.4:
                            if vec_6[0] >= 0:
                                vec6 = 5
                            elif vec_6[0] < 0:
                                vec6 = 1
                        elif slope_6 < -0.41:
                            if vec_6[0] >= 0:
                                vec6 = 4
                            elif vec_6[0] < 0:
                                vec6 = 8
                        elif slope_6 < 0.41:
                            if vec_6[1] >= 0:
                                vec6 = 3
                            elif vec_6[1] < 0:
                                vec6 = 7
                        elif slope_6 < 2.4:
                            if vec_6[1] >= 0:
                                vec6 = 2
                            elif vec_6[1] < 0:
                                vec6 = 6
                        else:
                            if vec_6[0] >= 0:
                                vec6 = 5
                            elif vec_6[0] < 0:
                                vec6 = 1

                        if slope_7 < -2.4:
                            if vec_7[0] >= 0:
                                vec7 = 5
                            elif vec_7[0] < 0:
                                vec7 = 1
                        elif slope_7 < -0.41:
                            if vec_7[0] >= 0:
                                vec7 = 4
                            elif vec_7[0] < 0:
                                vec7 = 8
                        elif slope_7 < 0.41:
                            if vec_7[1] >= 0:
                                vec7 = 3
                            elif vec_7[1] < 0:
                                vec7 = 7
                        elif slope_7 < 2.4:
                            if vec_7[1] >= 0:
                                vec7 = 2
                            elif vec_7[1] < 0:
                                vec7 = 6
                        else:
                            if vec_7[0] >= 0:
                                vec7 = 5
                            elif vec_7[0] < 0:
                                vec7 = 1

                        if slope_8 < -2.4:
                            if vec_8[0] >= 0:
                                vec8 = 5
                            elif vec_8[0] < 0:
                                vec8 = 1
                        elif slope_8 < -0.41:
                            if vec_8[0] >= 0:
                                vec8 = 4
                            elif vec_8[0] < 0:
                                vec8 = 8
                        elif slope_8 < 0.41:
                            if vec_8[1] >= 0:
                                vec8 = 3
                            elif vec_8[1] < 0:
                                vec8 = 7
                        elif slope_8 < 2.4:
                            if vec_8[1] >= 0:
                                vec8 = 2
                            elif vec_8[1] < 0:
                                vec8 = 6
                        else:
                            if vec_8[0] >= 0:
                                vec8 = 5
                            elif vec_8[0] < 0:
                                vec8 = 1

                        if slope_9 < -2.4:
                            if vec_9[0] >= 0:
                                vec9 = 5
                            elif vec_9[0] < 0:
                                vec9 = 1
                        elif slope_9 < -0.41:
                            if vec_9[0] >= 0:
                                vec9 = 4
                            elif vec_9[0] < 0:
                                vec9 = 8
                        elif slope_9 < 0.41:
                            if vec_9[1] >= 0:
                                vec9 = 3
                            elif vec_9[1] < 0:
                                vec9 = 7
                        elif slope_9 < 2.4:
                            if vec_9[1] >= 0:
                                vec9 = 2
                            elif vec_9[1] < 0:
                                vec9 = 6
                        else:
                            if vec_9[0] >= 0:
                                vec9 = 5
                            elif vec_9[0] < 0:
                                vec9 = 1

                        if slope_10 < -2.4:
                            if vec_10[0] >= 0:
                                vec10 = 5
                            elif vec_10[0] < 0:
                                vec10 = 1
                        elif slope_10 < -0.41:
                            if vec_10[0] >= 0:
                                vec10 = 4
                            elif vec_10[0] < 0:
                                vec10 = 8
                        elif slope_10 < 0.41:
                            if vec_10[1] >= 0:
                                vec10 = 3
                            elif vec_10[1] < 0:
                                vec10 = 7
                        elif slope_10 < 2.4:
                            if vec_10[1] >= 0:
                                vec10 = 2
                            elif vec_10[1] < 0:
                                vec10 = 6
                        else:
                            if vec_10[0] >= 0:
                                vec10 = 5
                            elif vec_10[0] < 0:
                                vec10 = 1

                        if slope_11 < -2.4:
                            if vec_11[0] >= 0:
                                vec11 = 5
                            elif vec_11[0] < 0:
                                vec11 = 1
                        elif slope_11 < -0.41:
                            if vec_11[0] >= 0:
                                vec11 = 4
                            elif vec_11[0] < 0:
                                vec11 = 8
                        elif slope_11 < 0.41:
                            if vec_11[1] >= 0:
                                vec11 = 3
                            elif vec_11[1] < 0:
                                vec11 = 7
                        elif slope_11 < 2.4:
                            if vec_11[1] >= 0:
                                vec11 = 2
                            elif vec_11[1] < 0:
                                vec11 = 6
                        else:
                            if vec_11[0] >= 0:
                                vec11 = 5
                            elif vec_10[0] < 0:
                                vec11 = 1

                        ####
                        self.game.stop("N",vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9, vec10, vec11)
                                              
                        return


        
        
        
    def timer_end(self):
        'タイマーを終了'

        if self.timer is not None:
            self.master.after_cancel(self.timer)
            self.timer = None

    def timer_start(self):
        'タイマーを開始'

        if self.timer is not None:
            # タイマーを一旦キャンセル
            self.master.after_cancel(self.timer)

        # テトリス実行中の場合のみタイマー開始
        if self.running:
            # タイマーを開始
            self.timer = self.master.after(1000, self.timer_event)

    def left_key_event(self, event):
        '左キー入力受付時の処理'

        # ブロックを左に動かす
        self.game.move_block(MOVE_LEFT)

    def right_key_event(self, event):
        '右キー入力受付時の処理'

        # ブロックを右に動かす
        self.game.move_block(MOVE_RIGHT)
        
    def space_key_event(self, event):
        'スペースキー入力受付時の処理'
        # ブロックを下に動かす
        self.game.move_block(MOVE_DOWN)
        print("space")
        # 落下タイマーを再スタート
        self.timer_start()
        
    def down_key_event(self, event):
        '下キー入力受付時の処理'

        # ブロックを90°回転する
        self.game.rot_block(MOVE_ROT)
        
    def up_key_event(self, event):
        '上キー入力受付時の処理'

        # ブロックを90°回転する
        self.game.rot_block(MOVE_ROT_INV)   

    def timer_event(self):
        'タイマー満期になった時の処理'

        # 下キー入力受付時と同じ処理を実行
        self.space_key_event(None)


class Application(tk.Tk):
    def __init__(self):
        super().__init__()

        # アプリウィンドウの設定
        self.geometry("400x690")
        self.title("人体テトリス")
        """
        label1 = tk.Label(self, text="Hello, World!")  #文字ラベル設定
        label1.pack(side="bottom")
        """
        # テトリス生成
        game = TetrisGame(self)
        """
        font2 = font.Font(family='Times', size=40)
        label3 = tk.Label(self, text='Score :　 ' + str(game.field.score), font=font2)
        label3.pack(side="bottom")
        """

        

        # イベントハンドラー生成
        EventHandller(self, game)

"""
def main():
    'main関数'
    # GUIアプリ生成
    app = Application()
    var.set("SCORE :  " + str(score))
    label = tk.Label(app, textvariable=var, width=30, bg="white")
    label.pack()
    #label.bind("<ButtonPress-1>", update_label)
    app.mainloop()
"""


'main関数'
# GUIアプリ生成
app = Application()

var_life = tk.StringVar()
var_life.set("★★★")
font = font.Font(family='Times', size=40)
label_life = tk.Label(app, textvariable=var_life, font=font)
label_life.pack(side="bottom")
label_life.pack()

var = tk.StringVar()
var.set("SCORE :  " + str(score))
#font = font.Font(family='Times', size=40)
label = tk.Label(app, textvariable=var, font=font)
label.pack(side="bottom")
label.pack()

var_gameover = tk.StringVar()
#font = font.Font(family='Times', size=40)
label_gameover = tk.Label(app, textvariable=var_gameover, font=font)
label_gameover.pack(side="bottom")
label_gameover.pack()

app.mainloop()
"""
if __name__ == "__main__":
    main()
"""