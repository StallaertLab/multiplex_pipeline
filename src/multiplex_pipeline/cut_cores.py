import os
import argparse
import  pandas as pd

def cut_cores(input_dir, df, output_dir, selection_list = None):
    '''
    Function to cut the cores from the image and save them as separate images.
    '''

    # read the df
    df = pd.read_pickle(df)

    # keep only the selected cores
    if selection_list:
        df = df[df['core_id'].isin(selection_list[0])]

    # get list of subdirectories
    dir_list = os.listdir(input_dir)

    df.apply(lambda x: x.core_name in dir_list, axis=1)



    return True

def main():
    
    # Create the argument parser
    parser = argparse.ArgumentParser()

    # Compulsory arguments
    parser.add_argument("input_dir", type=str, help="Path to the directory with the image and the core data.")
    parser.add_argument("output_dir", type=str, help="Path to the directory where the cores will be saved.")
    parser.add_argument("df", type=str, help="Path to the dataframe with the core data.")

    # Additional arguments
    parser.add_argument("-s", "--selection_list", type=str, help="List to the cores to be cut.")

    # Cut the cores
    args = parser.parse_args()

    result = cut_cores(args.input_dir, args.df, args.output_dir, args.selection_list)


if __name__ == "__main__":
    main()

