import argparse
import os

from scripts.convert_notebooks import execute_and_convert_notebooks_to_json
from scripts.generate_page_html import generate_page_html
from scripts.get_commit_hash import get_commit_hash


def get_page_paths(path=None):
    """
    Recursively get paths to all markdown files in a directory

    Parameters
    ----------
    path : (str | None)
        The root directory to search. If None, defaults to the "content" folder in the
        working directory. This parameter is used internally for recursion and should
        initially be called with None or excluded.

    Returns
    -------
    dict
        A dictionary mapping markdown page paths relative to the "content" directory
        to their absolute paths in the form of: { relative_path: absolute_path, ...}

        This may seem redundant at a first glance, but having the absolute paths as
        well aids greatly in producing the correct path links for local/dev builds
        where the absolute URL is not known

    Notes
    -----
    - README.md files are excluded.
    - Keys in the returned dictionary are paths relative to the "content" directory
    """

    md_pages = {}
    if path is None:
        path = os.path.join(
            os.getcwd(),
            "content",
        )
    directories = os.listdir(path)
    for item in directories:
        item_path = os.path.join(
            path,
            item,
        )
        if os.path.isdir(item_path):
            # add items from new dict into md_pages
            md_pages.update(
                get_page_paths(item_path),
            )
        else:
            if not item == "README.md" and item.endswith(".md"):
                rel_path = os.path.relpath(
                    item_path,
                    start=os.path.join(
                        os.getcwd(),
                        "content",
                    ),
                )
                md_pages[rel_path] = item_path

    return md_pages


def main():
    """
    Main function to generate html pages for deployment
    """

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
        "notebook_hashes.json",
    )

    commit_hash = get_commit_hash(
        build_on_dev_arg = args.build_on_dev,
    )

    # write_standalone_html=True,
    execute_and_convert_notebooks_to_json(
        input_folder=content_path,
        write_standalone_html=False,
        execute_notebooks=args.execute_notebooks,
        force_execute_all=args.force_execute_all,
        dev_build=commit_hash,
        nb_hash_path=nb_hash_path,
    )

    # AES TODO merge into generate_path_html or provide input like convert
    md_paths = get_page_paths()

    # generate_page_html(
    #     md_paths,
    #     dev_build=args.build_on_dev,
    # )


if __name__ == "__main__":
    main()
