import os
import os.path
import sys
import itertools

if __name__ == '__main__':

    models   = ["S30"] #["S30", "B30", "RS30", "RB30", "FTB30n", "FTB30p"] 
    attacks  = ["LinfAdditiveUniformNoise_attack"] # ["AutoAttack_adv"]

    SBATCH_FILE = "sbatch_experiments.sh"

    # Use itertools.product to generate the Cartesian product of all argument vectors
    count = 0
    for (
        model, attack,
        )  in  itertools.product(
        models, attacks
        ):

        # LAUNCH
        ####################
        cmd = "sbatch " + SBATCH_FILE
        cmd = cmd + " " + model
        cmd = cmd + " " + attack

        # Launch process
        print(cmd)
        os.system(cmd)

        count += 1

    print("All processes have been successfully launched")
    print("Total processes: %d" % count)


