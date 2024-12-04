import os
import argparse
from solver import Solver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)#test1: 0.001 test2:0.01 test3:0.1
    parser.add_argument("--batch_size", type=int, default=16)#test1: 32 test2:16
    parser.add_argument("--max_epochs", type=int, default=50)
    
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoint")
    parser.add_argument("--ckpt_name", type=str, default="depth")
    parser.add_argument("--evaluate_every", type=int, default=2)
    parser.add_argument("--visualize_every", type=int, default=100)
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join("C:\\", "Users","Davide","OneDrive - Università degli Studi di Perugia","DAVIDE - UNIVERSITA'","MAGISTRALE","2°ANNO","Deep_learning_&_Robot_perception","DepthEstimationUnreal"))

    parser.add_argument("--is_train", type=bool, default=False)
    parser.add_argument("--ckpt_file", type=str, default="depth_20.pth")

    args = parser.parse_args()
    solver = Solver(args)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    if args.is_train:
        solver.fit()
    else:
        solver.test()

if __name__ == "__main__":
    main()
