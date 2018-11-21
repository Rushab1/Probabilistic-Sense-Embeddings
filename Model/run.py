from train import *
import argparse

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-checkpoint_dir", type=str, default="../modelfiles/")
    args.add_argument("-modelfile", type=str, default="model.pkl")
    args.add_argument("-restart", type=str, default=0)
    args.add_argument("-num_batches", type=str, default=10)
    opts = args.parse_args()

    main(opts)
