import sys

from utils import Dataset


if __name__ == "__main__":
    ### READ ARGUMENTS ###
    try:
        input_file = sys.argv[1]
        aligner_mode = sys.argv[2]
        similarity_metric = sys.argv[3]
        output_file = sys.argv[4]#train or predict or eval
    except:
        raise ValueError("Incorrect commandline arguments")
### LOAD DATASET ###
    dataset = Dataset(input_file)
### RUN EVALUATION ###
    dataset.create_alignment_file(similarity_metric=similarity_metric,mode=aligner_mode,outfile=output_file)