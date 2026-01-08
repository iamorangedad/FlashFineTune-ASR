import pytest
import pytest_asyncio
import asyncio
import nats
import os
from nats.errors import TimeoutError

# --- 配置 ---
# 如果使用了 kubectl port-forward，通常是 localhost:4222
NATS_URL = os.getenv("NATS_URL", "nats://10.0.0.27:30742")

# --- Fixtures: 资源管理 ---


@pytest_asyncio.fixture
async def nats_client():
    """
    初始化 NATS 连接。
    每个测试用例都会获得一个独立的连接，测试结束后自动关闭。
    """
    # 1. 建立连接
    try:
        nc = await nats.connect(NATS_URL)
    except Exception as e:
        pytest.fail(f"无法连接 NATS 服务器: {e}")

    yield nc

    # 2. 清理资源 (Teardown)
    # 刷新缓冲区并关闭连接
    await nc.drain()


# --- 测试用例 ---


@pytest.mark.asyncio
async def test_basic_pub_sub(nats_client):
    """
    测试 1: 基础发布/订阅模式 (Fire and Forget)
    验证：发出的消息能被订阅者原样收到
    """
    topic = "tests.basic"
    payload = b"Hello NATS"

    # 创建一个 Future 对象来捕获异步回调的结果
    # 这是测试异步回调函数的标准做法
    future = asyncio.Future()

    async def message_handler(msg):
        # 当收到消息时，把结果放入 future，解除测试的阻塞
        if not future.done():
            future.set_result(msg.data)

    # 1. 订阅
    await nats_client.subscribe(topic, cb=message_handler)

    # 2. 发布
    await nats_client.publish(topic, payload)

    # 3. 断言
    try:
        # 等待 future 有结果，最多等 1 秒
        received_data = await asyncio.wait_for(future, timeout=1.0)
        assert received_data == payload
    except asyncio.TimeoutError:
        pytest.fail("测试失败：订阅者在 1 秒内未收到消息")


@pytest.mark.asyncio
async def test_request_reply(nats_client):
    """
    测试 2: 请求/响应模式 (Request-Reply)
    验证：类似 HTTP 的同步等待回执功能
    """
    topic = "tests.service"
    request_data = b"Can you help?"
    response_data = b"Yes, I can"

    # 模拟一个服务端：收到请求后，回复特定数据
    async def service_handler(msg):
        # msg.respond 直接回复给发送者
        await msg.respond(response_data)

    # 1. 启动服务订阅
    await nats_client.subscribe(topic, cb=service_handler)

    # 2. 发起请求 (Request)
    # request 方法会阻塞等待，直到收到回复或超时
    try:
        response = await nats_client.request(topic, request_data, timeout=1.0)

        # 3. 断言
        assert response.data == response_data
        print(f"\n[Pass] Got response: {response.data.decode()}")

    except TimeoutError:
        pytest.fail("测试失败：请求超时，未收到回复")


@pytest.mark.asyncio
async def test_queue_group(nats_client):
    """
    测试 3: 队列组 (Queue Group) - 负载均衡测试
    验证：发多条消息，应该被组内的订阅者分摊，而不是每个人都收到所有消息
    """
    topic = "tests.queue"
    queue_group_name = "workers"

    # 计数器
    received_count = 0

    async def worker_handler(msg):
        nonlocal received_count
        received_count += 1

    # 启动两个订阅者，属于同一个 Queue Group
    await nats_client.subscribe(topic, queue=queue_group_name, cb=worker_handler)
    await nats_client.subscribe(topic, queue=queue_group_name, cb=worker_handler)

    # 发送 10 条消息
    for i in range(10):
        await nats_client.publish(topic, f"msg-{i}".encode())

    # 给一点时间处理
    await asyncio.sleep(0.5)

    # 正常订阅模式下，2个订阅者 x 10条消息 = 应该收到 20 次
    # 但在 Queue Group 模式下，10条消息只会被处理 10 次（分摊）
    assert received_count == 10
