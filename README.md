# 环境

ubuntu18.04+ros-melodic

# 依赖

安装evo：

	$ pip install evo --upgrade --no-binary evo

说明：只有align.py需要evo

# 编译
	$ cd catkin_ws/src/

	$ git clone https://github.com/AIRS-TJ/view_tool.git
 
	$ cd ..

	$ catkin_make

# 运行

## 1.对齐两条轨迹

	$ cd src/view_tool/

	$ ./align.py data/1/fuse1811.bag /vrpn_client_node/bluerov/pose /odometry/filtered output.bag

说明：/vrpn_client_node/bluerov/pose 是对齐的参考轨迹，/odometry/filtered是要对齐的轨迹，参考轨迹放前面

data/1/fuse1811.bag是输入的bag，会在当前目录下输出output.bag

## 2.截取图像的主题

	$ rosbag filter data/1/2021-08-27-18-11all.bag output2.bag "topic=='/davis_left/image_raw'"

在当前目录下输出output2.bag

## 3.合并包

	$ ./merge_bag.py -v output3.bag output.bag output2.bag

将output.bag和output2.bag合并成output3.bag，在当前目录下输出output3.bag

## 4.显示

	$ roslaunch view_tool demo.launch 
