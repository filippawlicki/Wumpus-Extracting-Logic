from gymnasium.envs.registration import register

register(
    id='WumpusWorld-v0',
    entry_point='env.wumpus_world_env:WumpusWorldEnv',
)