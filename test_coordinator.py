from environments.game_coordinator import GameCoordinator
import numpy as np

coord = GameCoordinator(n_perceptive=2, seed=42)
obs = coord.reset()
print('Players:', len(obs))
print('Obs shape per agent:', obs[0].shape)

round_num = 0
while not coord.done and round_num < 200:
    actions = {
        i: coord.envs[i].action_space.sample()
        for i in coord.alive_players
    }
    if 0 in actions:
        actions[0][3] = 0.0
    obs, rewards, done, info = coord.step(actions)
    round_num += 1

print(f'Game ended: {info}')
print(f'Rounds: {round_num}')
print(f'Winner: {coord.winner}')
print('GameCoordinator OK')