# rpi-hailo-vision

sudo apt update
sudo apt install -y \
  python3-gi python3-gi-cairo \
  gir1.2-gtk-3.0 gir1.2-gtk-4.0 \
  gir1.2-gdkpixbuf-2.0 gir1.2-glib-2.0


python -V
#note the version number and change below appropriately
uv venv --python 3.12 --system-site-packages .venv


uv run detection-consolidated.py -i usb

