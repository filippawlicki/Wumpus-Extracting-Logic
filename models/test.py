from env.wumpus_world_env import WumpusWorldEnv
from models.dqn_agent import DQNAgent

env = WumpusWorldEnv(grid_size=4, default_map=False, num_of_pits=1)

obs, _ = env.reset()
done = False

state_dim = 10
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim, epsilon=0, epsilon2=0)  # No exploration

agent.load_model("checkpoints/model_ep10059_final.pt")

max_episodes = 1000
max_steps = 100
won_games = 0

episode = 0
print("Starting test...")

while episode < max_episodes:
    obs, _ = env.reset()
    done = False
    steps = 0

    while not done and steps < max_steps:
        action = agent.act(obs)
        if action is None:
            break

        obs, reward, done, truncated, info = env.step(action)
        if info["won"]:
            won_games += 1
            break
        steps += 1

    episode += 1

print(f"Test completed. Won {won_games}/{max_episodes} games. Win rate: {(won_games/max_episodes):.2%}")