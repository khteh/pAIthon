import requests,magic
from pathlib import Path
def download_file(file_url: str, local_file_path: Path) -> None:
    """Download a file and save it with the specified file name."""
    if not Path(local_file_path).exists() or not Path(local_file_path).is_file():
        response = requests.get(file_url)
        if response:
            local_file_path.write_bytes(response.content)
            print(f"File successfully downloaded and stored at: {local_file_path}")
        else:
            raise requests.exceptions.RequestException(
                f"Failed to download the file. Status code: {response.status_code}"
            )
    print(f"file magic: {magic.from_file(local_file_path)}")    