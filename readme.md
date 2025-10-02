> 这只是之前进组老师让我做的一个简单的小项目，用强化学习做一个简单的对连续体机器人的简单控制，没有什么学术价值，只是作为一个我的实践小项目做一下，效果也不算很好，权当给大家做一个参考。

依赖(ubuntu22 python3.10)

`
pip install stable-baselines3[extra] pybullet fpdf
`

运行训练：
`
python src/PPO_train.py
`

具体参数调整见[技术报告](https://blog.tosania.top/posts/rl-soft-first/)