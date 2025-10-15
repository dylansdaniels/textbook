import argparse
from pathlib import Path

from scripts.execute_and_convert_nbs import execute_and_convert_nbs_to_json
from scripts.generate_page_html import generate_page_html
from scripts.get_commit_hash import get_commit_hash

textbook_root_path = Path(__file__).parents[0]


def get_markdown_paths(root_path=None):
    """Recursively get paths to all markdown files in a directory (except READMEs)

    Parameters
    ----------
    root_path : (Path | str), default=None
        The root directory to search, assumed to be the top-level directory of your
        textbook, and assumed to have a "content" subdirectory. If None, defaults to the
        directory of this file.

    Returns
    -------
    dict
        A dictionary mapping markdown page paths relative to the "content" directory
        to their absolute paths in the form of:

        {
            relative_path: absolute_path,
            ...
        }

        This may seem redundant at a first glance, but having the absolute paths as
        well aids greatly in producing the correct path links for local/dev builds
        where the absolute URL is not known

    Notes
    -----
    - README.md files are excluded.
    """
    if not root_path:
        root_path = textbook_root_path

    content_path = Path(root_path / "content")
    # This glob is recursive, see https://docs.python.org/3/library/pathlib.html#pathlib-pattern-language
    paths_all = sorted(content_path.glob("**/*.md"))
    paths_readme_excluded = [p for p in paths_all if ("README" not in str(p))]
    md_paths = {
        str(p.relative_to(content_path)): str(p.absolute())
        for p in paths_readme_excluded
    }
    return md_paths


def main():
    """
    Main function to generate html pages for deployment

    AES TODO: describe required file structure
    """

    # AES TODO
    # replace the `dev_build` arg (which acts as both a flag and content) with 3 values. But what? not "version" since overloaded. Maybe just two? stable vs anything else? "variant". Actually build_type
    # - 4508128151
    # - stable

    # AES TODO simplify stack with pathlib.Path

    # accept command line arguments
    parser = argparse.ArgumentParser(
        description="Generate html pages for deployment",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--execution-filter",
        action="store",
        default="no-execution",
        choices=[
            "no-execution",
            "execute-updated-unskipped-notebooks",
            "execute-all-unskipped-notebooks",
            "execute-absolutely-all-notebooks",
        ],
        help="""Specify different criteria for which notebooks you want to execute before converting
them to HTML. The default is 'no-execution'. The four options are below, in order of
more execution:\n
- 'no-execution': This will not execute any notebooks. You may receive warnings if
  specific notebooks should be executed.\n
- 'execute-updated-unskipped-notebooks': Execute only notebooks which have been
  updated/changed or are new, excluding notebooks flagged for skipping.\n
- 'execute-all-unskipped-notebooks': Execute all notebooks except those flagged for
  skipping.\n
- 'execute-absolutely-all-notebooks': Execute all notebooks.
""",
    )
    parser.add_argument(
        "--build-on-dev",
        type=str,
        help="Optionally provide the commit from upstream/master",
    )
    # AES Not sure we want to support this, but leaving it as an option since I assume
    # this case was the reason why `os.getcwd()` is used so much in the scripts instead
    # of absolute paths.
    parser.add_argument(
        "--custom-root-path",
        type=str,
        help="Optionally provide a different 'root' location for your textbook files",
    )

    # add all above arguments to the parser
    args = parser.parse_args()

    if args.custom_root_path:
        root_path = args.custom_root_path
    else:
        root_path = textbook_root_path

    content_path = Path(root_path / "content")
    nb_hash_path = Path(root_path / "scripts" / "nb_hashes.json")
    nb_skip_path = Path(root_path / "scripts" / "nbs_to_skip.json")

    print(
        f"Configuration: Choosing notebooks based on '--execution-filter={args.execution_filter}'"
    )

    # AES ref output to "build_type"
    commit_hash = get_commit_hash(build_on_dev_arg=args.build_on_dev)

    # AES commented out standalone
    # write_standalone_html=True,
    execute_and_convert_nbs_to_json(
        input_folder=content_path,
        write_standalone_html=False,
        dev_build=commit_hash,
        nb_hash_path=nb_hash_path,
        nb_skip_path=nb_skip_path,
        execution_filter=args.execution_filter,
    )

    # AES TODO move into generate_page_html since it only needs the content path input like the convert function above
    md_paths = get_markdown_paths()

    generate_page_html(
        md_paths,
        dev_build=args.build_on_dev,
    )


if __name__ == "__main__":
    main()
