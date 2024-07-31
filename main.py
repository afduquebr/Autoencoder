#######################################################################################################
"""
Created on Apr 19 2024

@author: Andres Felipe DUQUE BRAN
"""
#######################################################################################################

import getopt
import sys

def parse_args():
    dataset = None  # Variable to store the selected dataset type
    anomaly = None  # Variable to store the anomaly percentage

    try:
        # Parse command-line options and arguments
        opts, args = getopt.getopt(sys.argv[1:], "d:a:", ["dataset=", "anomaly="])
    except getopt.GetoptError as err:
        # Print error and exit if the command-line arguments are invalid
        print(str(err))
        sys.exit(2)

    # Process each option and argument
    for opt, arg in opts:
        if opt in ("-d", "--dataset"):
            # Check if the dataset argument is valid
            if arg not in ["sig", "bbox1", "bbox2", None]:
                raise ValueError("Dataset must be either 'sig', 'bbox1' or 'bbox2'.")
            dataset = arg  # Assign the valid dataset argument to the dataset variable
        elif opt in ("-a", "--anomaly"):
            # Check if the anomaly argument is provided
            if arg != None:
                try:
                    anomaly = float(arg)  # Try to convert the argument to a float
                except ValueError:
                    raise ValueError("Anomaly percentage must be a float.")

    # Check if the anomaly is specified and dataset is not None
    if anomaly == None and dataset != None:
        raise ValueError("Anomaly percentage option is required")

    return dataset, anomaly  # Return the parsed arguments

def main():
    try:
        # Call the parse_args function and get the dataset and anomaly values
        dataset, anomaly = parse_args()
        # Print the selected dataset and anomaly percentage
        print("Signal Dataset: ", dataset)
        print("Anomaly percentage: ", anomaly)
    except ValueError as err:
        # Print error message if ValueError is raised and exit the program
        print("Error:", err)
        sys.exit(1)

if __name__ == "__main__":
    # Entry point of the script; call the main function
    main()
