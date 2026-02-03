"""
Helper script to download Mona Lisa image
"""
import urllib.request
import os

def download_mona_lisa_image():
    """Download Mona Lisa image from Wikipedia"""
    url = "https://upload.wikimedia.org/wikipedia/commons/e/ec/Mona_Lisa%2C_by_Leonardo_da_Vinci%2C_from_C2RMF_retouched.jpg"
    filename = "mona_lisa.jpg"
    
    if os.path.exists(filename):
        print(f"{filename} already exists. Skipping download.")
        return filename
    
    try:
        print(f"Downloading Mona Lisa image from Wikipedia...")
        urllib.request.urlretrieve(url, filename)
        print(f"Successfully downloaded {filename}")
        return filename
    except Exception as e:
        print(f"Error downloading image: {e}")
        print("\nPlease manually download a Mona Lisa image and save it as 'mona_lisa.jpg'")
        print("You can use this URL:")
        print(url)
        return None

if __name__ == "__main__":
    download_mona_lisa_image()
