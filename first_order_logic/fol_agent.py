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
        list(self.prolog.query("retractall(start(_, _))"))
        list(self.prolog.query("retractall(orientation(_, _))"))
        list(self.prolog.query("retractall(kb(_, _, _, _))"))
        list(self.prolog.query("retractall(result(_, _))"))
        list(self.prolog.query("retractall(grid_size(_))"))
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

        # Add starting position
        sx, sy = self.env.entrance
        self.prolog.assertz(f"start({sx}, {sy})")

        size = self.env.grid_size
        self.prolog.assertz(f"grid_size({size})")

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

        obs, _, _, _, feedback = self.env.step(action)

        return obs, feedback["won"]

    def render_environment(self):
        self.env.render()
        time.sleep(0.35)
        # input("Press Enter to continue...")  # Wait for user input after rendering

    def run(self):
        done = False

        for i in range(self.max_steps):
            action, has_won, _ = self.act()
            if has_won:
                done = True
                break

        return 1 if done else 0

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

        obs, has_won = self.execute_action(action_int)

        self.update_kb(obs)
        self.state_counter += 1
        self.update_agent_position_and_orientation()

        return action_int, has_won, obs


if __name__ == "__main__":
    rendering = False
    default_map = False
    log = False
    GAME_COUNT = 10_000
    winCount = 0
    for _ in range(GAME_COUNT) if not rendering else range(1):
        env = WumpusWorldEnv(default_map=default_map, num_of_pits=3)
        agent = FOLAgent(env, rendering=rendering, log=log)
        win = agent.run()
        winCount += win
    if not rendering:
        print(f"{GAME_COUNT} games finished. Win rate: {winCount/GAME_COUNT * 100:.2f}%")
    else:
        print(f"Game finished.")