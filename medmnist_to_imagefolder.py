# medmnist_to_imagefolder.py
from pathlib import Path
from PIL import Image
import importlib
from medmnist import INFO

SUBSET = "bloodmnist"  # 可改为 DermaMNIST / BloodMNIST / OCTMNIST / OrganAMNIST 等
OUTDIR = Path.cwd()/ "medmnist_data" / SUBSET

def export_split(dataset_cls, split, outdir, as_rgb=True):
    ds = dataset_cls(split=split, download=True, as_rgb=as_rgb)
    imgs, labels = ds.imgs, ds.labels.squeeze()
    sd = outdir / split
    for i, (img, y) in enumerate(zip(imgs, labels)):
        d = sd / str(int(y))
        d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(d / f"{i}.png")

def main():
    info = INFO[SUBSET.lower()]
    mod = importlib.import_module("medmnist")
    dataset_cls = getattr(mod, info["python_class"])
    for split in ["train", "val", "test"]:
        export_split(dataset_cls, split, OUTDIR, as_rgb=True)
    print("OK:", OUTDIR)

if __name__ == "__main__":
    main()
# comparason.py