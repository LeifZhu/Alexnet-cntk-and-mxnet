import os
import argparse


def generate_map_file(abs_data_dir):
    train_dir = os.path.join(abs_data_dir, "train")
    val_dir = os.path.join(abs_data_dir, "validation")
    class_dirs = os.listdir(train_dir)
    
    with open(os.path.join(abs_data_dir, "train_map.txt"), "w") as f:
        for i in range(len(class_dirs)):
            pic_dirname = os.path.join(train_dir, class_dirs[i])
            for pic in os.scandir(pic_dirname):
                if pic.is_file() and pic.name.endswith('.JPEG'):
                    f.write("%s\t%d\n"%(os.path.join(pic_dirname,pic),i))
                    
    with open(os.path.join(abs_data_dir, "val_map.txt"), "w") as f:
        for i in range(len(class_dirs)):
            pic_dirname = os.path.join(val_dir, class_dirs[i])
            for pic in os.scandir(pic_dirname):
                 if pic.is_file() and pic.name.endswith('.JPEG'):
                    f.write("%s\t%d\n"%(os.path.join(pic_dirname,pic),i))
    
    print("generate mapfile successfully!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "generate mapfile")
    parser.add_argument("--data_dir", type=str, default="/home/leizhu/dataset/imagenet8/raw/",
                    help="absolute path of data directory")
    args = parser.parse_args()
    generate_map_file(args.data_dir)
