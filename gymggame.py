#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: zhixiang time:18-4-2

import gym
env = gym.make('MountainCar-v0')
env.reset()
for _ in range(100000):
    env.render()
    env.step(env.action_spample()) # take a random action