import getopt
import sys

def parse_args():
    path = None
    dataset = None
    anomaly = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], "p:d:a:", ["path=", "dataset=", "anomaly="])
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-p", "--path"):
            if arg not in ["local", "server", None]:
                raise ValueError("Path must be either 'local' or 'server'.")
            path = arg
        elif opt in ("-d", "--dataset"):
            if arg not in ["sig1", "sig2", "bbox", None]:
                raise ValueError("Dataset must be either 'sig1', 'sig2' or 'bbox'.")
            dataset = arg
        elif opt in ("-a", "--anomaly"):
            if arg != None:
                try:
                    anomaly = float(arg)
                except ValueError:
                    raise ValueError("Anomaly percentage must be between 0 and 0.1.")

    if path == None:
        raise ValueError("Path option is required")
    if anomaly == None and dataset != None:
        raise ValueError("Anomaly percentage option is required")

    return path, dataset, anomaly

def main():
    try:
        path, dataset, anomaly = parse_args()
        print("Path: ", path)
        print("Signal Dataset: ", dataset)
        print("Anomaly percentage: ", anomaly)
    except ValueError as err:
        print("Error:", err)
        sys.exit(1)

if __name__ == "__main__":
    main()
