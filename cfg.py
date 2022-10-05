import torch

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

content_path="images/content.jpg"
style_path="images/style.jpg"

weight=1.0
output="out.png"
ospace="uniform"
resize_to=512