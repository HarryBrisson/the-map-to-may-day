import json
import shutil
from pathlib import Path
from typing import Any, Protocol


class JsonStorage(Protocol):
    def write_json(self, relative_path: str, data: Any) -> str:
        ...

    def read_json(self, relative_path: str) -> Any:
        ...

    def write_text(self, relative_path: str, data: str, content_type: str = "text/plain; charset=utf-8") -> str:
        ...

    def read_text(self, relative_path: str) -> str:
        ...

    def exists(self, relative_path: str) -> bool:
        ...

    def clear_prefix(self, relative_path: str) -> int:
        ...


class LocalJsonStorage:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir.resolve()

    def write_json(self, relative_path: str, data: Any) -> str:
        path = self._resolve(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return str(path)

    def read_json(self, relative_path: str) -> Any:
        path = self._resolve(relative_path)
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def write_text(self, relative_path: str, data: str, content_type: str = "text/plain; charset=utf-8") -> str:
        del content_type
        path = self._resolve(relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data, encoding="utf-8")
        return str(path)

    def read_text(self, relative_path: str) -> str:
        return self._resolve(relative_path).read_text(encoding="utf-8")

    def exists(self, relative_path: str) -> bool:
        return self._resolve(relative_path).exists()

    def clear_prefix(self, relative_path: str) -> int:
        path = self._resolve(relative_path)
        if not path.exists():
            return 0
        if path.is_file():
            path.unlink()
            return 1

        count = sum(1 for child in path.rglob("*") if child.is_file())
        shutil.rmtree(path)
        return count

    def _resolve(self, relative_path: str) -> Path:
        path = (self.data_dir / relative_path).resolve()
        try:
            path.relative_to(self.data_dir)
        except ValueError as exc:
            raise ValueError(f"Refusing to write outside data root: {relative_path}") from exc
        return path


class S3JsonStorage:
    def __init__(self, bucket: str, prefix: str = "") -> None:
        if not bucket:
            raise ValueError("S3 bucket is required for S3 output")
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("boto3 is required for S3 output") from exc

        self.client = boto3.client("s3")
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    def write_json(self, relative_path: str, data: Any) -> str:
        key = self._key(relative_path)
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body, ContentType="application/json")
        return f"s3://{self.bucket}/{key}"

    def read_json(self, relative_path: str) -> Any:
        response = self.client.get_object(Bucket=self.bucket, Key=self._key(relative_path))
        return json.loads(response["Body"].read().decode("utf-8"))

    def write_text(self, relative_path: str, data: str, content_type: str = "text/plain; charset=utf-8") -> str:
        key = self._key(relative_path)
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data.encode("utf-8"), ContentType=content_type)
        return f"s3://{self.bucket}/{key}"

    def read_text(self, relative_path: str) -> str:
        response = self.client.get_object(Bucket=self.bucket, Key=self._key(relative_path))
        return response["Body"].read().decode("utf-8")

    def exists(self, relative_path: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(relative_path))
            return True
        except Exception:
            return False

    def clear_prefix(self, relative_path: str) -> int:
        prefix = self._key(relative_path).rstrip("/") + "/"
        deleted = 0
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
            if not objects:
                continue
            self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": objects})
            deleted += len(objects)
        return deleted

    def _key(self, relative_path: str) -> str:
        clean_path = relative_path.strip("/")
        return f"{self.prefix}/{clean_path}" if self.prefix else clean_path


def get_user_path(user_email: str, stage: str, domain: str, provider: str, year: str, file_name: str) -> str:
    return f"user_data/{user_email}/{stage}/{domain}/{provider}/{year}/{file_name}"


def make_storage(output: str, data_dir: Path, s3_bucket: str = "", s3_prefix: str = "") -> JsonStorage:
    if output == "local":
        return LocalJsonStorage(data_dir)
    if output == "s3":
        return S3JsonStorage(bucket=s3_bucket, prefix=s3_prefix)
    raise ValueError(f"Unsupported output backend: {output}")
