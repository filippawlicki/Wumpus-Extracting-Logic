from pyswip import Prolog
import time

import config
from env.wumpus_world_env import WumpusWorldEnv

class FOLAgent:
    def __init__(self, env):
        self.env = env
        self.prolog = Prolog()
        self.prolog.consult(str(config.ROOT_DIR / "first_order_logic" / "logic.pl").replace("\\", "/"))
        self.state_counter = 0

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

    def update_kb(self):
        ax, ay = self.env.agent_pos
        perceptions = self.env._get_observation()
        perception_list = []
        if perceptions[0]:
            perception_list.append("stench")
        if perceptions[1]:
            perception_list.append("breeze")
        if perceptions[2]:
            perception_list.append("glitter")
        list(self.prolog.query(f"update_kb({ax}, {ay}, {perception_list})"))

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
        if action == "grab":
            if self.env.agent_pos == self.env.gold_pos:
                list(self.prolog.query(f"retract(gold({self.env.gold_pos[0]}, {self.env.gold_pos[1]}))"))  # Remove gold from Prolog
            self.env.step(3)  # ACTION_GRAB
        elif action == "climb":
            self.env.step(4)  # ACTION_CLIMB
        elif action == "move":
            self.env.step(0)  # ACTION_MOVE_FORWARD
        elif action == "turn_left":
            self.env.step(1)  # ACTION_TURN_LEFT
        elif action == "turn_right":
            self.env.step(2)  # ACTION_TURN_RIGHT
        elif action == "shoot":
            self.env.step(5)  # ACTION_SHOOT

    def render_environment(self):
        self.env.render()
        time.sleep(0.5)  # Wait for 1 second after rendering
        # input("Press Enter to continue...")  # Wait for user input after rendering

    def run(self):
        # Initialize Prolog with environment data
        self.initialize_prolog()

        # Loop until the game ends
        done = False
        while not done:
            # Render the environment at each step
            self.render_environment()

            # Update knowledge base
            self.update_kb()

            # Determine the next action using Prolog
            action = self.make_decision()

            # Update the result in Prolog
            self.update_result(action)

            # Execute the action in the environment
            self.execute_action(action)

            # Increment the state counter
            self.state_counter += 1

            # Update agent position and orientation in Prolog
            self.update_agent_position_and_orientation()

            # Check if the game is done
            if action == "climb" and self.env.agent_pos == self.env.entrance:
                done = True

if __name__ == "__main__":
    env = WumpusWorldEnv()
    agent = FOLAgent(env)
    agent.run()
