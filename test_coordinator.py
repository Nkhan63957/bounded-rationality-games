from environments.game_coordinator import GameCoordinator
from agents.rule_based_regular import RuleBasedRegular
import numpy as np

coord = GameCoordinator(n_perceptive=2, seed=42)
obs = coord.reset()
print('Players:', len(obs))
print('Obs shape per agent:', obs[0].shape)

# Build rule-based agents for regular players
rule_based = {}
for i in range(1, 20):
    rule_based[i] = RuleBasedRegular(
        player_id=i,
        neighborhood=coord.envs[0].neighborhoods[i],
        seed=42 + i,
    )

round_num = 0
while not coord.done and round_num < 100:
    actions = {}

    # Jack: random (no PPO yet)
    jack_action = coord.envs[0].action_space.sample()
    jack_action[6] = 0.0  # disable concession
    actions[0] = jack_action

    # Regular players: rule-based
    env_state = {
        "alive_mask":    coord.envs[0].alive_mask,
        "symbols":       coord.envs[0].symbols,
        "player_states": coord.envs[0].players,
        "round":         coord.round,
    }
    for i in coord.alive_players:
        if i != 0:
            actions[i] = rule_based[i].act(obs[i], env_state)

    obs, rewards, done, info = coord.step(actions)
    round_num += 1

print(f'Game ended: {info}')
print(f'Rounds: {round_num}')
print(f'Winner: {coord.winner}')
print('Rule-based test OK')