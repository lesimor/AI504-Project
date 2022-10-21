import gdown

google_path = "https://drive.google.com/uc?id="
file_id = "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
output_name = "img_align_celeba.zip"
gdown.download(google_path + file_id, output_name, quiet=False)
