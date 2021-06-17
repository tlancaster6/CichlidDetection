import argparse, os

"""primary command line executable script."""

# example usage
# python3 CichlidDetection.py

# parse command line arguments
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='Available Commands', dest='command')

download_parser = subparsers.add_parser('download')

annotatate_parser = subparsers.add_parser('annotate')

train_parser = subparsers.add_parser('train')
train_parser.add_argument('-e', '--Epochs', type=int, default=10, help='number of epochs to train')

sync_parser = subparsers.add_parser('sync')

detect_parser = subparsers.add_parser('detect')
detect_parser.add_argument('-t', '--Test', action='store_true', help='run detection on 10 images from the test set')
detect_parser.add_argument('-p', '--PIDs', type=str, help="project ID's to analyze")
detect_parser.add_argument('-c', '--Continuous', action='store_true', help='continuously check for and run detection on images in each of the specified projects until told to stop')

args = parser.parse_args()

# determine the absolute path to the directory containing this script, and the host name
package_root = os.path.dirname(os.path.abspath(__file__))

if args.command == 'sync':
    from Modules.FileManager import FileManager
    FileManager().sync_training_dir()

else:
    from Modules.Runner import Runner
    runner = Runner()

    if args.command == 'annotate':
        runner.annotate()

    if args.command == 'download':
        runner.download()

    elif args.command == 'train':
        runner.prep()
        runner.train(num_epochs=args.Epochs)

    elif args.command == 'detect':
        if not args.Continuous:
            for pid in args.PIDs:
                runner.detect(pid)


