import argparse
import os
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
        str(p.relative_to(content_path)): str(p.absolute()) for p in paths_readme_excluded
    }
    return md_paths


def main():
    """
    Main function to generate html pages for deployment
    """

    # AES TODO
    # replace the `dev_build` arg (which acts as both a flag and content) with 3 values. But what? not "version" since overloaded. Maybe just two? stable vs anything else? "variant". Actually build_type
    # - 4508128151
    # - stable

    # AES TODO simplify stack with pathlib.Path

    # accept command line arguments
    parser = argparse.ArgumentParser(description="Generate html pages for deployment")
    parser.add_argument(
        "--execute-notebooks",
        action="store_true",
        help="Execute notebooks as needed based on their status "
        "before converting them to HTML.",
    )
    parser.add_argument(
        "--force-execute-all",
        action="store_true",
        help="Force execute all notebooks regardless of their status",
    )
    parser.add_argument(
        "--build-on-dev",
        type=str,
        help="Optionally provide the commit from upstream/master",
    )

    # add all above arguments to the parser
    args = parser.parse_args()

    content_path = os.path.join(
        os.getcwd(),
        "content",
    )
    nb_hash_path = os.path.join(
        os.getcwd(),
        "scripts",
        "nb_hashes.json",
    )

    # AES ref output to "build_type"
    commit_hash = get_commit_hash(build_on_dev_arg=args.build_on_dev)

    # write_standalone_html=True,
    execute_and_convert_nbs_to_json(
        input_folder=content_path,
        write_standalone_html=False,
        execute_nbs=args.execute_notebooks,
        force_execute_all=args.force_execute_all,
        dev_build=commit_hash,
        nb_hash_path=nb_hash_path,
    )

    # AES TODO move into generate_page_html
    md_paths = get_markdown_paths()

    generate_page_html(
        md_paths,
        dev_build=args.build_on_dev,
    )


if __name__ == "__main__":
    main()
