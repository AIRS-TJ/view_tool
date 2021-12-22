# 环境

ubuntu18.04+ros-melodic

2021-12-20-11-21-24.bag在网盘/Dataset/airsbot 路径上

# 依赖

安装evo：

	$ pip install evo --upgrade --no-binary evo


# 运行

	$ cd src/view_tool/

	$  ./align.py ~/airsbot_ws/2021-12-20-11-21-24.bag /vrpn_client_node/airsbot/pose /tracked_pose /reach_goal
