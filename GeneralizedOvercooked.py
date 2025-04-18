import random
import cv2

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from AutonomousSystemProject.utils import load_and_delete_img


rew_shaping_params = {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5,
        "DISH_DISP_DISTANCE_REW": 0,
        "POT_DISTANCE_REW": 0,
        "SOUP_DISTANCE_REW": 0,
    }


# AGENT 0 IS THE GREEN ONE
class GeneralizedOvercooked:
    def __init__(self, layouts, info_level=0, horizon=400, use_r_shaped=False, old_dynamics=False):

        self.envs = []
        for layout in layouts:
            if use_r_shaped:
                base_mdp = OvercookedGridworld.from_layout_name(layout, rew_shaping_params=rew_shaping_params, old_dynamics=old_dynamics)
            else:
                base_mdp = OvercookedGridworld.from_layout_name(layout, old_dynamics=old_dynamics)

            base_env = OvercookedEnv.from_mdp(base_mdp, info_level=info_level, horizon=horizon)
            env = Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)
            self.envs.append(env)
        self.cur_env = self.envs[0]
        self.observation_space, self.action_space = self.cur_env.observation_space, self.cur_env.action_space

    def reset(self):
        idx = random.randint(0, len(self.envs)-1)
        self.cur_env = self.envs[idx]
        return self.cur_env.reset()

    def step(self, *args):
        return self.cur_env.step(*args)

    def render(self, state, timeout=500):
        state_img = self.get_state_img(state)
        cv2.imshow("State", state_img)
        cv2.moveWindow("State", 100, 2000)
        cv2.waitKey(timeout)

    def get_state_img(self, state):
        img_path = self.cur_env.visualizer.display_rendered_state(state, grid=self.cur_env.base_env.mdp.terrain_mtx, img_path="state_img.png")
        return load_and_delete_img(img_path)


