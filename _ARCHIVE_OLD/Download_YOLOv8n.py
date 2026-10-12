import urllib.request
import hashlib
from pathlib import Path

URL = "https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.pt?download=true"

OUT = Path(r"D:\Doan\yolov8n.pt")

SHA256_EXPECTED = (
    "31e20dde3def09e2cf938c7be6fe23d9150bbbe503982af13345706515f2ef95"
)


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


print("Dang tai YOLOv8n...")
print("Nguon: Hugging Face - Ultralytics")
print("Luu tai:", OUT)

request = urllib.request.Request(
    URL,
    headers={
        "User-Agent": "Mozilla/5.0"
    }
)

with urllib.request.urlopen(request, timeout=120) as response:

    total = response.headers.get("Content-Length")

    if total:
        total = int(total)

    downloaded = 0

    with open(OUT, "wb") as f:

        while True:

            data = response.read(256 * 1024)

            if not data:
                break

            f.write(data)

            downloaded += len(data)

            if total:

                percent = downloaded / total * 100

                print(
                    f"\rDa tai: {downloaded / 1024 / 1024:.2f} MB "
                    f"/ {total / 1024 / 1024:.2f} MB "
                    f"({percent:.1f}%)",
                    end=""
                )

            else:

                print(
                    f"\rDa tai: {downloaded / 1024 / 1024:.2f} MB",
                    end=""
                )

print("\nTai xong.")

hash_value = sha256_file(OUT)

print("SHA256:", hash_value)

if hash_value == SHA256_EXPECTED:

    print("KIEM TRA OK")
    print("MODEL SAN SANG:")
    print(OUT)

else:

    print("LOI: SHA256 khong dung.")
    print("Khong nen su dung file vua tai.")