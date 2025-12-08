import boto3
import pytest
import os
import uuid
from botocore.client import Config
from io import BytesIO

# --- 配置部分 ---
# 你可以修改这里，或者通过环境变量传入
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "http://10.0.0.27:39001")
ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")  # 请替换为你的实际账号
SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "password123")  # 请替换为你的实际密码


@pytest.fixture(scope="module")
def s3_client():
    """
    初始化 boto3 客户端，连接到 MinIO
    """
    client = boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",  # MinIO 默认忽略区域，但 boto3 需要填一个
    )
    return client


@pytest.fixture(scope="function")
def test_bucket(s3_client):
    """
    Fixture:
    1. 创建一个随机名字的临时 Bucket
    2. 将 Bucket 名字传给测试函数
    3. 测试结束后，自动清空并删除 Bucket (Teardown)
    """
    bucket_name = f"test-bucket-{uuid.uuid4().hex}"

    # 1. 创建 Bucket
    try:
        s3_client.create_bucket(Bucket=bucket_name)
        print(f"\n[Setup] Created bucket: {bucket_name}")
    except Exception as e:
        pytest.fail(f"无法创建 Bucket: {e}")

    yield bucket_name

    # 3. 清理工作 (Teardown)
    print(f"\n[Teardown] Cleaning up bucket: {bucket_name}")
    try:
        # 必须先删除桶内所有对象才能删除桶
        objects = s3_client.list_objects_v2(Bucket=bucket_name)
        if "Contents" in objects:
            for obj in objects["Contents"]:
                s3_client.delete_object(Bucket=bucket_name, Key=obj["Key"])

        # 删除空桶
        s3_client.delete_bucket(Bucket=bucket_name)
    except Exception as e:
        print(f"清理 Bucket 失败: {e}")


def test_upload_and_download(s3_client, test_bucket):
    """
    核心测试逻辑：
    1. 上传一段文本
    2. 下载这段文本
    3. 比对内容是否一致
    """
    file_name = "test_data.txt"
    content_to_upload = b"Hello MinIO! This is a test string."

    # --- 1. 上传 (Upload) ---
    print(f"[*] Uploading {file_name}...")
    try:
        s3_client.put_object(
            Bucket=test_bucket,
            Key=file_name,
            Body=content_to_upload,
            ContentType="text/plain",
        )
    except Exception as e:
        pytest.fail(f"上传失败: {e}")

    # --- 2. 验证元数据 (Optional) ---
    # 确认文件真的存在
    try:
        s3_client.head_object(Bucket=test_bucket, Key=file_name)
    except Exception:
        pytest.fail("文件上传后无法找到 (HeadObject 失败)")

    # --- 3. 下载 (Download) ---
    print(f"[*] Downloading {file_name}...")
    try:
        response = s3_client.get_object(Bucket=test_bucket, Key=file_name)
        downloaded_content = response["Body"].read()
    except Exception as e:
        pytest.fail(f"下载失败: {e}")

    # --- 4. 断言 (Assert) ---
    assert (
        downloaded_content == content_to_upload
    ), f"内容不匹配! 期望: {content_to_upload}, 实际: {downloaded_content}"

    print("[Pass] Upload and Download check successful.")


def test_list_objects(s3_client, test_bucket):
    """
    额外测试：测试列出文件功能
    """
    s3_client.put_object(Bucket=test_bucket, Key="file1.txt", Body=b"1")
    s3_client.put_object(Bucket=test_bucket, Key="file2.txt", Body=b"2")

    response = s3_client.list_objects_v2(Bucket=test_bucket)

    # 断言里面应该有 2 个文件
    assert response["KeyCount"] == 2
    keys = [obj["Key"] for obj in response["Contents"]]
    assert "file1.txt" in keys
    assert "file2.txt" in keys
