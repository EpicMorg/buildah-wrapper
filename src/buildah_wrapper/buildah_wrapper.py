import sys

from buildah_wrapper.settings import SCRIPT_VERSION
from buildah_wrapper.helper._dataclass import ArgParser, BuildahBuildError
from buildah_wrapper.helper.class_buildah import BuildahBuilder
from buildah_wrapper.helper.log_print import logger, show_help, show_version


def main():
    """Parse arguments and drive build / deploy / clean via BuildahBuilder."""
    parser = ArgParser()
    args = parser.parse_args()

    if args.log_level:
        logger.setLevel(args.log_level)
    elif args.verbose:
        logger.setLevel("DEBUG")

    if args.help or len(sys.argv) == 1:
        show_help()
        return

    if args.version:
        show_version()
        return

    if not (args.build or args.deploy or args.clean):
        # No action requested: show help rather than silently doing nothing.
        show_help()
        return

    try:
        builder = BuildahBuilder(args)
        builder.run()
    except BuildahBuildError as e:
        # One or more builds/pushes returned a non-zero exit code.
        logger.error(f"Operation failed: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Invalid value: {e}")
        sys.exit(1)
    except RuntimeError as e:
        # preflight: buildah missing or not runnable.
        logger.error(f"Preflight failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
