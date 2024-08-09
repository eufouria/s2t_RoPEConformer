import subprocess

urls = [
    "https://us.openslr.org/resources/12/train-clean-100.tar.gz",
    "https://us.openslr.org/resources/12/train-clean-360.tar.gz",
    "https://us.openslr.org/resources/12/train-other-500.tar.gz",
    "https://us.openslr.org/resources/12/dev-clean.tar.gz",
    "https://us.openslr.org/resources/12/dev-other.tar.gz",
    "https://us.openslr.org/resources/12/test-clean.tar.gz",
    "https://us.openslr.org/resources/12/test-other.tar.gz"
]

destination = "data/"

subprocess.run(["mkdir", "data"])
subprocess.run(["mkdir", "checkpoints"])
for url in urls:
    subprocess.run(["wget", "-P", destination, url])
