import getopt
import sys

def parse_args():
    path = None
    scale = None
    middle = None
    latent = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], "p:s:m:l:", ["path=","scale=", "middle=", "latent="])
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-p", "--path"):
            if arg not in ["local", "server"]:
                raise ValueError("Scaling must be either 'local' or 'server'.")
            path = arg
        elif opt in ("-s", "--scale"):
            if arg not in ["minmax", "standard"]:
                raise ValueError("Scaling must be either 'minmax' or 'standard'.")
            scale = arg
        elif opt in ("-m", "--middle"):
            try:
                middle = int(arg)
            except ValueError:
                raise ValueError("Middle layer dimension must be an integer.")
        elif opt in ("-l", "--latent"):
            try:
                latent = int(arg)
            except ValueError:
                raise ValueError("Latent space dimension must be an integer.")

    if path is None:
        raise ValueError("Path option is required")
    if scale is None:
        raise ValueError("Scale option is required")
    if middle is None:
        raise ValueError("Middle option is required")
    if latent is None:
        raise ValueError("Latent option is required")

    return path, scale, middle, latent

def main():
    try:
        path, model, middle, latent = parse_args()
        print("Path: ", path)
        print("Model: ", model)
        print("Middle layer dimension: ", middle)
        print("Latent space dimension: ", latent)
    except ValueError as err:
        print("Error:", err)
        sys.exit(1)

if __name__ == "__main__":
    main()
