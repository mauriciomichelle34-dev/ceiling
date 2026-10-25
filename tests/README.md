# 测试

安装项目依赖后，从项目根目录运行：

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

84 项测试覆盖坐标方向、PLC 数据传输、接图自动处理、相机接口及 PLC 到位自动拍照。测试使用模拟相机、内存中的 PLC 替身和本机回环 S7 服务，不访问真实设备。HTTP 测试需要允许 Python 访问本机临时端口。

`s7_test_server.py` 仅为测试补充 python-snap7 3.3.0 模拟服务端的位写入语义，不修改生产客户端。模拟测试不能替代真实相机、图像标定及 PLC 动作时序的联调。
