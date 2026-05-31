"""
Quick sanity check for PBeautyContest environment.
Verifies: reset, step, reward computation, implied level calculation.
Run before writing any agent code.
"""

from environments.p_beauty_contest import PBeautyContest
import numpy as np

print("=" * 50)
print("PBeautyContest Environment Sanity Check")
print("=" * 50)

# Test 1: Basic initialization
env = PBeautyContest(n_players=5, p=2/3,
                     reward_type="winner_take_all",
                     opponent_type="mixed")
obs, _ = env.reset()
print(f"\nTest 1 — Initialization:")
print(f"  Observation shape: {obs.shape} (expected (4,))")
print(f"  Action space: {env.action_space}")
print(f"  Observation space: {env.observation_space}")

# Test 2: Level-k computation
print(f"\nTest 2 — Level-k submissions (p=2/3):")
for k, v in env._level_k.items():
    if k <= 5:
        print(f"  Level-{k}: {v:.4f}")
print(f"  Nash equilibrium: 0.0")
print(f"  Human mean (Nagel 1995): ~24.9")

# Test 3: Step through one round
action = np.array([33.3], dtype=np.float32)  # Level-1 play
obs, reward, terminated, truncated, info = env.step(action)
print(f"\nTest 3 — Single step (submitting {action[0]:.1f}):")
print(f"  Mean submission: {info['mean']:.3f}")
print(f"  Target:          {info['target']:.3f}")
print(f"  Reward:          {reward:.3f}")
print(f"  Implied level:   {info['implied_level']:.3f}")

# Test 4: Implied level accuracy
print(f"\nTest 4 — Implied level computation:")
test_cases = [
    (50.0, 0.0, "Level-0 (random)"),
    (33.3, 1.0, "Level-1"),
    (22.2, 2.0, "Level-2"),
    (14.8, 3.0, "Level-3"),
    (0.0,  10.0, "Nash equilibrium"),
]
for submission, expected, label in test_cases:
    implied = env._implied_level(submission)
    status = "✓" if abs(implied - expected) < 0.5 else "✗"
    print(f"  {status} {label}: submission={submission:.1f} "
          f"implied={implied:.2f} expected≈{expected:.1f}")

# Test 5: Run full episode
print(f"\nTest 5 — Full episode (100 rounds, proximity reward):")
env2 = PBeautyContest(n_players=5, p=2/3,
                      reward_type="proximity",
                      opponent_type="mixed")
obs, _ = env2.reset()
total_reward = 0.0
done = False
while not done:
    action = env2.action_space.sample()
    obs, reward, done, _, info = env2.step(action)
    total_reward += reward

df = env2.get_history_df()
print(f"  Total reward (random agent): {total_reward:.3f}")
print(f"  Mean submission:             {df['own_submission'].mean():.3f}")
print(f"  Mean implied level:          {df['implied_level'].mean():.3f}")
print(f"  Episode length:              {len(df)} rounds")

print(f"\n{'=' * 50}")
print("All tests complete. Environment is ready.")
print("=" * 50)