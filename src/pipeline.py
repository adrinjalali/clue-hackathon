import sys
import pandas as pd

from pre_process import process_level2, load_binary


def run():
    data = load_binary()
    print(data['users'].head())
    exit()

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
    run()
