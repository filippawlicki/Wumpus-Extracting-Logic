import numpy as np
from matplotlib import pyplot as plt
from pyswip import Prolog
import time

import config
from env.wumpus_world_env import WumpusWorldEnv

class FOLAgent:
    def __init__(self, env, rendering=False, log=False):
        self.env = env
        self.prolog = Prolog()
        self.prolog.consult(str(config.ROOT_DIR / "first_order_logic" / "logic.pl").replace("\\", "/"))
        self.state_counter = 0
        self.rendering = rendering
        self.log = log
        self.max_steps = 100

        self.initialize_prolog()

    def reset(self):
        self.state_counter = 0
        obs, _ = self.env.reset()
        self.initialize_prolog()
        return obs

    def initialize_prolog(self):
        # Clear existing facts in Prolog
        list(self.prolog.query("retractall(wumpus(_, _))"))
        list(self.prolog.query("retractall(pit(_, _))"))
        list(self.prolog.query("retractall(gold(_, _))"))
        list(self.prolog.query("retractall(agent(_, _, _))"))
        list(self.prolog.query("retractall(orientation(_, _))"))
        list(self.prolog.query("retractall(kb(_, _, _, _))"))
        list(self.prolog.query("retractall(result(_, _))"))
        list(self.prolog.query("retractall(log(_))"))

        # Add Wumpus position
        wx, wy = self.env.wumpus_pos
        self.prolog.assertz(f"wumpus({wx}, {wy})")

        # Add pits positions
        for px, py in self.env.pit_pos:
            self.prolog.assertz(f"pit({px}, {py})")

        # Add gold position
        gx, gy = self.env.gold_pos
        self.prolog.assertz(f"gold({gx}, {gy})")

        # Add agent position and orientation
        ax, ay = self.env.agent_pos
        self.prolog.assertz(f"agent({ax}, {ay}, {self.state_counter})")
        self.prolog.assertz(f"orientation({['north', 'east', 'south', 'west'][self.env.agent_dir]}, {self.state_counter})")

        size = self.env.grid_size

        # Initialize knowledge base
        list(self.prolog.query(f"initialize_kb({size})"))

        if self.log:
            self.prolog.assertz(f"log(true)")

        # Update knowledge base
        self.update_kb(self.env._get_observation())

    def update_kb(self, perceptions):
        ax, ay = self.env.agent_pos
        # perceptions = self.env._get_observation()
        perception_list = []
        # print(perceptions)
        if perceptions[0]:
            perception_list.append("stench")
        if perceptions[1]:
            perception_list.append("breeze")
        if perceptions[2]:
            perception_list.append("glitter")
        if perceptions[3]:
            perception_list.append("bump")
        if perceptions[4]:
            perception_list.append("scream")
        if perceptions[5]:
            perception_list.append("hasgold")
        if perceptions[6]:
            perception_list.append("on_entrance")
        list(self.prolog.query(f"update_kb({ax}, {ay}, {perception_list}, {self.state_counter})"))

    def update_result(self, action):
        # Assert the result of the current action in Prolog
        self.prolog.assertz(f"result({action}, {self.state_counter})")

    def update_agent_position_and_orientation(self):
        ax, ay = self.env.agent_pos
        direction = ['north', 'east', 'south', 'west'][self.env.agent_dir]
        self.prolog.assertz(f"agent({ax}, {ay}, {self.state_counter})")
        self.prolog.assertz(f"orientation({direction}, {self.state_counter})")

    def make_decision(self):
        # Query Prolog for the best decision with state awareness
        decision = list(self.prolog.query(f"make_decision(Action, {self.state_counter})"))[0]["Action"]
        return decision

    def execute_action(self, action):
        if action == config.ACTION_GRAB:
            if self.env.agent_pos == self.env.gold_pos:
                list(self.prolog.query(f"retract(gold({self.env.gold_pos[0]}, {self.env.gold_pos[1]}))"))
        elif action == config.ACTION_SHOOT:
            list(self.prolog.query(f"retract(wumpus({self.env.wumpus_pos[0]}, {self.env.wumpus_pos[1]}))"))

        obs, _, _, _, info = self.env.step(action)

        return obs, info

    def render_environment(self):
        self.env.render()
        time.sleep(0.35)
        # input("Press Enter to continue...")  # Wait for user input after rendering

    def run(self):
        for i in range(self.max_steps):
            action, info, _ = self.act()
            if info["won"]:
                break

        return info

    def act(self):
        # Render the environment at each step
        if self.rendering:
            self.render_environment()

        action = self.make_decision()
        self.update_result(action)

        # Convert action to integer for execution
        action_map = {
            "move": config.ACTION_MOVE_FORWARD,
            "turn_left": config.ACTION_TURN_LEFT,
            "turn_right": config.ACTION_TURN_RIGHT,
            "grab": config.ACTION_GRAB,
            "climb": config.ACTION_CLIMB,
            "shoot": config.ACTION_SHOOT
        }
        action_int = action_map[action]

        obs, info = self.execute_action(action_int)

        self.update_kb(obs)
        self.state_counter += 1
        self.update_agent_position_and_orientation()

        return action_int, info, obs


def save_steps_plot(steps_to_win):
    """ Save a histogram showing the distribution of steps needed to win. """
    plt.figure(figsize=(12, 6))

    min_steps = min(steps_to_win)
    max_steps = max(steps_to_win)

    bins = np.arange(min_steps, max(steps_to_win) + 2) - 0.5
    plt.hist(steps_to_win, bins=bins, color='blue', alpha=0.7, rwidth=0.85)

    plt.xlabel('Number of Steps to Win')
    plt.ylabel('Number of Games')
    plt.title('Distribution of Steps Needed to Win')

    step = max(1, (max_steps - min_steps) // 20)
    plt.xticks(np.arange(min_steps, max(steps_to_win) + 1, step=step))
    plt.tight_layout()
    plt.savefig(f"steps_to_win_distribution.png")
    plt.close()

if __name__ == "__main__":
    rendering = False
    default_map = False
    testing = True
    log = False
    GAME_COUNT = 5_000
    checkpoint_interval = 100
    winCount = 0
    deadCount = 0
    notPossibleCount = 0
    steps_to_win = []
    env = WumpusWorldEnv(default_map=default_map, num_of_pits=3)

    for i in range(GAME_COUNT) if testing else range(1):
        _, info = env.reset()
        if not info["possible_to_win"]:
            notPossibleCount += 1
            continue
        agent = FOLAgent(env, rendering=rendering, log=log)
        info = agent.run()
        if info["won"]:
            steps_to_win.append(agent.state_counter)
            winCount += 1
        if info["dead"]:
            deadCount += 1

        if testing and i % checkpoint_interval == 0:
            print(f"Game {i + 1}/{GAME_COUNT} completed. Won: {winCount}, Dead: {deadCount}, Possible to win: {notPossibleCount}")
    if testing:
        print(f"Test completed. Won {winCount}/{GAME_COUNT} games. Win rate: {winCount/GAME_COUNT * 100:.2f}%")
        print(f"Survived {GAME_COUNT - deadCount}/{GAME_COUNT} games. Survival rate: {(GAME_COUNT - deadCount) / GAME_COUNT:.2%}")
        print(f"Not possible to win in {notPossibleCount}/{GAME_COUNT} games. Not possible rate: {(notPossibleCount / GAME_COUNT):.2%}")

        if steps_to_win:
            save_steps_plot(steps_to_win)
        else:
            print("No games were won, no steps to plot.")
    else:
        print(f"Game finished.")