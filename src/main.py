import argparse


## TO DO ##
# Automate the different phases of the LA prediction

# 1 : Preprocessing
# Creation of the files with all the necessary data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-chr", "--chronicle", type=int, required=True)
    parser.add_argument("-approx", "--approximation", type=int, required=False)
    parser.add_argument("-rate", "--rate", type=float, required=False)
    parser.add_argument("-site", "--sitenumber", type=int, required=True)
    parser.add_argument("-ref", "--ref", action="store_true")
    parser.add_argument("-step", "--step", type=int, required=False)
    parser.add_argument("-f", "--folder", type=str, required=True)
    parser.add_argument("-sd", "--steady", type=int, required=False)
    parser.add_argument("-s", "--s", action="store_true")
    parser.add_argument("-v", "--v", action="store_true")
    parser.add_argument("-perm", "--permeability", type=float, required=False)

    args = parser.parse_args()

    approx = args.approximation
    chronicle = args.chronicle
    site_number = args.sitenumber
    rate = args.rate
    ref = args.ref
    folder = args.folder
    perm = args.permanent
    sat = args.s
    vul = args.v
    perm = args.permeability
