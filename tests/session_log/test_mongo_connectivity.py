import pytest
import os
import uuid
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

# --- Fixtures (前置/后置处理) ---


@pytest.fixture(scope="module")
def mongo_uri():
    """
    获取连接地址。
    优先读取环境变量 MONGO_URI，如果没有则默认为 localhost (用于 port-forward 测试)
    """
    return os.getenv("MONGO_URI", "http://10.0.0.27:27017")


@pytest.fixture(scope="module")
def mongo_client(mongo_uri):
    """
    建立数据库连接，测试结束后自动断开。
    scope="module" 表示整个测试文件只建立一次连接，提高效率。
    """
    # 设置 3 秒连接超时，避免连不上时一直卡死
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)

    yield client  # 测试开始前运行到这里

    # --- Teardown (测试结束后清理) ---
    try:
        # 清理测试创建的临时数据库
        client.drop_database("pytest_check_db")
    except Exception:
        pass
    client.close()


@pytest.fixture(scope="function")
def test_collection(mongo_client):
    """
    为每个测试函数提供一个干净的集合 (Collection)。
    """
    db = mongo_client["pytest_check_db"]
    collection = db["smoke_test"]

    # 每次测试前清空集合
    collection.delete_many({})

    yield collection

    # 测试后也可以选择清空
    collection.delete_many({})


# --- Test Cases (测试用例) ---


def test_connection_ping(mongo_client):
    """
    测试 1: 验证能否 Ping 通服务器 (最基础连接)
    """
    try:
        # admin command ping 是开销最小的检查方式
        mongo_client.admin.command("ping")
    except ConnectionFailure as e:
        pytest.fail(f"无法连接到 MongoDB, 请检查端口转发或网络地址。错误: {e}")


def test_write_permission(test_collection):
    """
    测试 2: 验证写入权限 (解决 Permission Denied 问题的关键验证)
    """
    sample_doc = {
        "_id": "write_check_001",
        "service": "asr-service",
        "status": "active",
    }

    try:
        result = test_collection.insert_one(sample_doc)
        assert result.inserted_id == "write_check_001"
    except OperationFailure as e:
        pytest.fail(f"写入失败，可能是 /data 目录权限不足: {e}")


def test_read_consistency(test_collection):
    """
    测试 3: 验证写入的数据能否被读回 (数据完整性)
    """
    # 1. 写入
    unique_id = str(uuid.uuid4())
    payload = {"_id": unique_id, "content": "hello pytest"}
    test_collection.insert_one(payload)

    # 2. 读取
    fetched_doc = test_collection.find_one({"_id": unique_id})

    # 3. 断言
    assert fetched_doc is not None, "未找到刚才写入的数据"
    assert fetched_doc["content"] == "hello pytest"


def test_server_info(mongo_client):
    """
    测试 4: 打印服务器版本信息 (可选，用于确认版本是否正确降级)
    """
    server_info = mongo_client.server_info()
    version = server_info.get("version")
    print(f"\n[Info] MongoDB Version: {version}")

    # 如果你要求必须是 4.4.x
    assert version.startswith("4.4"), f"版本不符合预期! 当前版本: {version}"
