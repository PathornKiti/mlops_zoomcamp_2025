import os
import requests
import argparse

def download_file(url, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
        print(f"Created folder: {dest_folder}")
    
    local_filename = os.path.join(dest_folder, url.split("/")[-1])
    print(f"Downloading {url} -> {local_filename}")
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    print(f"Downloaded: {local_filename}")

def main():
    parser = argparse.ArgumentParser(description="Download taxi trip data files.")
    parser.add_argument("--color", type=str, required=True, help="Color of the taxi (e.g., green, yellow)")
    parser.add_argument("--year", type=int, required=True, help="Year of the data (e.g., 2023)")
    parser.add_argument("--months", type=int, nargs="+", required=True, help="List of months to download (e.g., 1 2 3)")
    parser.add_argument("--dest_folder", type=str, default="homework2/data", help="Destination folder for downloaded files")
    
    args = parser.parse_args()

    for month in args.months:
        month_str = f"{month:02d}"  # ensures leading zero (e.g., 01, 02)
        url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{args.color}_tripdata_{args.year}-{month_str}.parquet"
        download_file(url, args.dest_folder)

if __name__ == "__main__":
    main()
