from openpi.shared import download

ck_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")

print(ck_dir)
