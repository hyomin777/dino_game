import numpy as np
from PIL import Image

import cv2
import io
import time
import random
import pickle
import os
from io import BytesIO
import base64
import json
import pandas as pd

from collections import deque
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.keys import Keys

from webdriver_manager.chrome import ChromeDriverManager

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


GAME_URL = "chrome://dino"
CHROME_DRIVER_PATH = ChromeDriverManager().install()

DATA_DIR = "./data"
MODEL_DIR = "./model"
SAVE_INTERVAL = 1000

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

PARAMS_FILE = os.path.join(DATA_DIR, "params.pkl")

INIT_SCRIPT = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"
GET_BASE64_SCRIPT = "canvasRunner = document.getElementById('runner-canvas'); return canvasRunner.toDataURL().substring(22)"


def save_params(params):
    with open(PARAMS_FILE, 'wb') as f:
        pickle.dump(params, f, pickle.HIGHEST_PROTOCOL)


def load_params():
    if os.path.isfile(PARAMS_FILE):
        with open(PARAMS_FILE, 'rb') as f:
            return pickle.load(f)
    return {
        "D": deque(maxlen=50000),
        "time": 0,
        "epsilon": 0.01
    }


def load_model(model):
    if os.path.isfile('./latest.pth'):
        model.load_state_dict(torch.load('./latest.pth'))
    return model


def grab_screen(driver):
    image_b64 = driver.execute_script(GET_BASE64_SCRIPT)
    screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
    return process_img(screen)


def process_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image[:300, :500]
    image = cv2.resize(image, (80, 80))
    return image


def show_img(graphs=False):
    while True:
        screen = (yield)
        window_title = "logs" if graphs else "game_play"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        imS = cv2.resize(screen, (800, 400))
        cv2.imshow(window_title, screen)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


class Game:
    def __init__(self):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        service = Service(CHROME_DRIVER_PATH)
        self._driver = webdriver.Chrome(
            service=service, options=chrome_options)
        self._driver.set_window_position(x=300, y=300)
        self._driver.set_window_size(900, 600)

        try:
            self._driver.get(GAME_URL)
        except:
            pass

        self._driver.execute_script("Runner.config.ACCELERATION=0")
        self._driver.execute_script(INIT_SCRIPT)

    def get_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")

    def get_playing(self):
        return self._driver.execute_script("return Runner.instance_.playing")

    def restart(self):
        self._driver.execute_script("Runner.instance_.restart()")

    def press_up(self):
        self._driver.find_element("tag name", "body").send_keys(Keys.ARROW_UP)

    def press_down(self):
        self._driver.find_element(
            "tag name", "body").send_keys(Keys.ARROW_DOWN)

    def get_score(self):
        score_array = self._driver.execute_script(
            "return Runner.instance_.distanceMeter.digits")
        return int(''.join(score_array))

    def pause(self):
        self._driver.execute_script("return Runner.instance_.stop()")

    def resume(self):
        self._driver.execute_script("return Runner.instance_.play()")

    def end(self):
        self._driver.close()


class DinoAgent:
    def __init__(self, game):
        self._game = game
        self.jump()

    def is_running(self):
        return self._game.get_playing()

    def is_crashed(self):
        return self._game.get_crashed()

    def jump(self):
        self._game.press_up()

    def duck(self):
        self._game.press_down()


class GameState:
    def __init__(self, agent, game):
        self._agent = agent
        self._game = game
        self._display = show_img()
        self._display.__next__()

    def get_state(self, actions):
        score = self._game.get_score()
        reward = 0.1
        is_over = False

        if actions[1] == 1:
            self._agent.jump()
            reward = -0.01

        image = grab_screen(self._game._driver)
        self._display.send(image)

        if self._agent.is_crashed():
            self._game.restart()
            reward = -10
            is_over = True

        return image, reward, is_over


ACTIONS = 2
GAMMA = 0.99
OBSERVATION = 1000
EXPLORE = 500000
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.1
REPLAY_MEMORY = 100000
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
IMG_CHANNELS = 4


class DinoNet(nn.Module):
    def __init__(self):
        super(DinoNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, (8, 8), stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, (4, 4), stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, (3, 3), stride=1, padding=1)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d((2, 2))
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.max_pool2d(self.relu(self.conv3(x)))
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_network(model, game_state, observe=False):
    params = load_params()
    D = params["D"]
    t = params["time"]
    epsilon = params["epsilon"]

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    do_nothing = np.zeros(2)
    do_nothing[0] = 1

    x_t, r_0, terminal = game_state.get_state(do_nothing)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])

    OBSERVE = 999999999 if observe else 100

    while True:
        loss_sum = 0
        a_t = np.zeros([2])

        if random.random() <= epsilon:
            action_index = random.randrange(2)
            a_t[action_index] = 1
        else:
            q = model(torch.tensor(s_t).float())
            _, action_index = torch.max(q, 1)
            action_index = action_index.item()
            a_t[action_index] = 1

        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        x_t1, r_t, terminal = game_state.get_state(a_t)
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        if len(D) > 50000:
            D.pop()
        D.append((s_t, action_index, r_t, s_t1, terminal))

        if t > OBSERVE:
            minibatch = random.sample(D, 16)
            inputs = np.zeros(
                (16, s_t.shape[1], s_t.shape[2], s_t.shape[3]))
            targets = np.zeros((16, 2))

            for i in range(16):
                state_t, action_t, reward_t, state_t1, terminal = minibatch[i]
                inputs[i:i + 1] = state_t
                target = model(torch.tensor(state_t).float()
                            ).detach().numpy()[0]
                Q_sa = model(torch.tensor(state_t1).float()
                            ).detach().numpy()[0]

                if terminal:
                    target[action_t] = reward_t
                else:
                    target[action_t] = reward_t + 0.99 * np.max(Q_sa)

                targets[i] = target

            outputs = model(torch.tensor(inputs).float())
            loss = loss_fn(outputs, torch.tensor(targets).float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        s_t = s_t1 if not terminal else s_t
        t += 1

        if t % SAVE_INTERVAL == 0:
            game_state._game.pause()
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"episode_{t}.pth"))
            torch.save(model.state_dict(), "./latest.pth")
            save_params({"D": D, "time": t, "epsilon": epsilon})
            game_state._game.resume()

        print(f'timestep: {t}, epsilon: {round(epsilon, 3)}, action: {action_index}, reward: {r_t}, loss: {round(loss_sum, 3)}')



def play_game(observe=False):
    params = {"D": deque(maxlen=50000), "time": 0, "epsilon": 0.001}
    save_params(params)
    game = Game()
    agent = DinoAgent(game)
    game_state = GameState(agent, game)
    try:
        model = DinoNet()
        model = load_model(model)
        train_network(model, game_state, observe)
    except StopIteration:
        game.end()
        
        
play_game(observe=False)