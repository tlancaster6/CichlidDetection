import argparse, os

"""primary command line executable script."""

# parse command line arguments
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='Available Commands', dest='command')

annotate_parser = subparsers.add_parser('annotate')
annotate_parser.add_argument('-p', '--PIDs', type=str, help="project ID's to annotate", required=True)
annotate_parser.add_argument('-n', '--Number', type=int, default=10, help='Limit annotation to x number of frames per pid')
annotate_parser.add_argument('-d', '--Dry', action='store_true', help='practice annotating without saving results')

train_parser = subparsers.add_parser('train')
train_parser.add_argument('-e', '--Epochs', type=int, default=10, help='number of epochs to train')
train_parser.add_argument('-d', '--Dry', action='store_true', help='skip uploading training results to cloud')

sync_parser = subparsers.add_parser('sync')

detect_parser = subparsers.add_parser('detect')
detect_parser.add_argument('-p', '--PIDs', type=str, help="project ID's to analyze", required=True)
detect_parser.add_argument('-c', '--Continuous', action='store_true', help='continuously check for and run detection on images in each of the specified projects until told to stop')

args = parser.parse_args()

# determine the absolute path to the directory containing this script, and the host name
package_root = os.path.dirname(os.path.abspath(__file__))

if args.command == 'sync':
    from Modules.FileManager import FileManager
    fm = FileManager(training=True)
    fm.sync_training_dir()
    fm.sync_model_dir()

else:
    from Modules.Runner import Runner
    if args.command == 'train':
        runner = Runner(training=True)
        runner.download_all()
        runner.prep()
        runner.train(num_epochs=args.Epochs, upload_results=(not args.Dry))
    elif args.command == 'annotate':
        runner = Runner(training=False, pids=args.PIDs)
        runner.annotate(dry=args.Dry)
    elif args.command == 'detect':
        runner = Runner(training=False, pids=args.PIDs)
        if args.Continuous:
            runner.continuous_detect()
        else:
            runner.detect()




