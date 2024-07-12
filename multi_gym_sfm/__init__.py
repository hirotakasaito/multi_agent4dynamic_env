from gymnasium.envs.registration import register

register(
    id = 'multi_gym_sfm-v0',
    entry_point = 'multi_gym_sfm.envs:GymSFM'
)
