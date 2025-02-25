import requests,magic,zipfile
from pathlib import Path
def Download(file_url: str, local_file_path: Path) -> None:
    """Download a file and save it with the specified file name."""
    if not local_file_path.exists() or not local_file_path.is_file():
        response = requests.get(file_url)
        if response:
            local_file_path.write_bytes(response.content)
            print(f"File successfully downloaded and stored at: {local_file_path}")
        else:
            raise requests.exceptions.RequestException(
                f"Failed to download the file. Status code: {response.status_code}"
            )
    print(f"file magic: {magic.from_file(local_file_path)}")

def Unzip(local_file_path: Path, destination) -> None:
    if local_file_path.exists() and local_file_path.is_file():
        with zipfile.ZipFile(local_file_path, 'r') as zf:
            zf.extractall(destination)

def Rename(local_file_path: Path, newname) -> None:
    if local_file_path.exists() and not Path(newname).exists():
        local_file_path.rename(newname)