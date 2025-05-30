import config
from env.wumpus_world_env import WumpusWorldEnv
from models.dqn_agent import DQNAgent

env = WumpusWorldEnv(grid_size=4, default_map=False, num_of_pits=3)

obs, _ = env.reset()
done = False

state_dim = config.STATE_DIM
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim, epsilon=0, epsilon2=0, min_epsilon=0, min_epsilon2=0)  # No exploration
agent.load_model("checkpoints/model_ep20000.pt")

max_episodes = 5000
max_steps = 100
won_games = 0
dead_games = 0
not_possible_games = 0

episode = 0
print("Starting test...")

while episode < max_episodes:
    obs, info = env.reset()
    possible_to_win = info["possible_to_win"]
    if not possible_to_win:
        not_possible_games += 1
        continue
    done = False
    steps = 0

    while not done and steps < max_steps:
        action = agent.act(obs)
        if action is None:
            break

        obs, reward, done, truncated, info = env.step(action)
        if info["dead"]:
            dead_games += 1
            break
        if info["won"]:
            won_games += 1
            break
        steps += 1

    episode += 1

print(f"Test completed. Won {won_games}/{max_episodes} games. Win rate: {(won_games/max_episodes):.2%}")
print(f"Survived {max_episodes - dead_games}/{max_episodes} games. Survival rate: {(max_episodes - dead_games)/max_episodes:.2%}")
print(f"Not possible to win in {not_possible_games}/{max_episodes} games. Not possible rate: {(not_possible_games/max_episodes):.2%}")