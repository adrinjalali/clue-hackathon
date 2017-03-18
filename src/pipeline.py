import sys
import pandas as pd

from src.pre_process import process_level2, load_binary


def run(data_fname):

    data = load_binary(data_fname)

    # (Opt: user clustering)

    # Extract features
    processed_data = process_level2()

    # (Opt: Select features)
    pass

    # Pre-predict processing
    pass

    # Predict
    pass


    # Post-predict processing
    pass



    # save results in the correct format.
    results.to_csv('./result.txt', index=None)


if __name__ == '__main__':
    print('run training')
    data_fname = sys.argv[-1]
    run(data_fname)
