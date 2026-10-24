# PLC、自动流程与相机接口测试

先安装项目根目录 `requirements.txt` 中的依赖，再从项目根目录运行：

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

测试使用内存中的 PLC 替身、本机回环 S7 服务和模拟相机，不连接真实 PLC 或相机。HTTP 测试需要允许 Python 访问本机临时端口。`s7_test_server.py` 只为测试补充 python-snap7 3.3.0 模拟服务端的位写入语义，不修改生产客户端。

2026-10-24 验证：64 项测试通过。真实相机另已完成连接和 5472×3648 取图验证；实拍全黑的问题仍待排查，详见根目录 `相机拍照使用说明.md`。
