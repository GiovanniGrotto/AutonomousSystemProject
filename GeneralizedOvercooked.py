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


# AGENT 0 IS CHOSEN RANDOMLY
class GeneralizedOvercooked:
    def __init__(self, layouts, info_level=0, horizon=400, use_r_shaped=False, old_dynamics=False, curriculum_goal=None):
        self.curriculum_goal = curriculum_goal
        self.total_steps = 1
        self.envs = []
        self.curriculum_level = 0
        self.curriculum_ended = False

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

    def reset(self, avg_rew=None):
        curriculum_promotion = False
        if avg_rew and self.curriculum_goal:  # This means we are inside the curriculum learning
            if avg_rew > self.curriculum_goal:
                self.curriculum_level = min(len(self.envs), self.curriculum_level+1)
                if self.curriculum_level < len(self.envs):
                    curriculum_promotion = True
                    print()
                    print(f"Updated curriculum level to: {self.curriculum_level}")
                    print(f"New layout: {self.envs[self.curriculum_level].base_env.mdp.layout_name}")
            idx = self.curriculum_level
            if self.curriculum_level == len(self.envs):
                if not self.curriculum_ended:
                    curriculum_promotion = True
                self.curriculum_ended = True
                # Here we have reached the reward goal in each env, so now sample equally
                idx = random.randint(0, len(self.envs) - 1)
        else:
            idx = random.randint(0, len(self.envs)-1)
        self.cur_env = self.envs[idx]
        return self.cur_env.reset(), curriculum_promotion

    def step(self, *args):
        self.total_steps += 1
        return self.cur_env.step(*args)

    def render(self, state, timeout=500):
        state_img = self.get_state_img(state)
        cv2.imshow("State", state_img)
        cv2.moveWindow("State", 100, 2000)
        cv2.waitKey(timeout)

    def get_state_img(self, state):
        img_path = self.cur_env.visualizer.display_rendered_state(state, grid=self.cur_env.base_env.mdp.terrain_mtx, img_path="state_img.png")
        return load_and_delete_img(img_path)


