import argparse
import os
from pathlib import Path
import subprocess

import requests
from hnn_core import __version__ as installed_hnn_version

from scripts.convert_notebooks import convert_notebooks_to_html
from scripts.generate_pages import generate_page_html

textbook_root_path = Path(__file__).parents[0]


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
    hash_path = os.path.join(
        os.getcwd(),
        "scripts",
        "notebook_hashes.json",
    )

    # get the version of hnn installed in the environment
    # this is needed for the checks below
    try:
        installed_hnn_commit = subprocess.check_output(["pip", "freeze"], text=True)
        for line in installed_hnn_commit.splitlines():
            if "hnn" in line:
                if "@" in line:
                    installed_hnn_commit = line.split("@")[2].split("#")[0]
                else:
                    installed_hnn_commit = line.split("hnn-core==")[-1]
        print(
            "\nThe installed version of hnn-core being used for this "
            f"build is:\n   {installed_hnn_commit}"
        )

    except Exception as e:
        raise RuntimeError(
            f"Could not import hnn_core and retrieve the latest commit:\n{e}"
        )

    if args.build_on_dev is not None:
        if args.build_on_dev == "master":
            # get the latest commit from upstream/master
            url = (
                "https://api.github.com/repos/jonescompneurolab/hnn-core/commits/master"
            )
            response = requests.get(url)
            response.raise_for_status()
            commit_hash = response.json()["sha"]
            if commit_hash != installed_hnn_commit:
                raise RuntimeError(
                    f"The latest commit on master ({commit_hash}) "
                    "does not match the latest commit on the installed "
                    f"version of hnn-core ({installed_hnn_commit})."
                    "\n"
                    "Try creating an environment by running the following commands "
                    "in a terminal:"
                    "\nmake create-textbook-dev-build"
                    "\nconda activate textbook-dev-build"
                )
        else:
            repo_hash = args.build_on_dev.strip()
            try:
                repo, commit = repo_hash.split(":")

                url = f"https://api.github.com/repos/{repo}/hnn-core/commits/{commit}"
                response = requests.get(url)
                response.raise_for_status()
                commit_hash = response.json()["sha"]

            except Exception as e:
                raise RuntimeError(
                    "the --dev-version argument must be specified in the "
                    'format: --dev-version "your-repository:your-commit-hash" '
                    "\nE.g., a valid input would be: jonescompneurolab:9e14b99"
                    f"\n\nError message: {e}"
                )

            if commit_hash != installed_hnn_commit:
                raise RuntimeError(
                    "The repository and commit you specified: "
                    f"\n   Repository: {repo}"
                    f"\n   Commit: {commit_hash} "
                    "\nDo not match the latest commit on the installed "
                    "version of hnn-core: "
                    f"\n   Installed version / commit: {installed_hnn_commit}"
                    "\nPlease ensure you have installed the proper version of "
                    "hnn-core in your local environment."
                    "\nTry creating an environment by running the following "
                    "commands in a terminal:"
                    "\n   $ make create-textbook-dev-build"
                    "\n   $ conda activate textbook-dev-build"
                    "\n   $ pip install --upgrade --force-reinstall --no-cache-dir "
                    f'"hnn-core[dev] @ git+https://github.com/{repo}/hnn-core.git@{commit}"'
                )
    else:
        commit_hash = False

        latest_stable = requests.get("https://pypi.org/pypi/hnn-core/json").json()[
            "info"
        ]["version"]

        if installed_hnn_version > latest_stable:
            print(
                "Warning: your installed version of hnn-core is ahead of the "
                "current stable version, but you did not use the --build-on-dev "
                "flag:"
                f"\n   Stable version: {latest_stable}"
                f"\n   Installed version: {installed_hnn_version}"
                "\nIt is generally advisable to use the --build-on-dev flag "
                "when generating the textbook on versions of hnn-core that are "
                "ahead of the current stable version."
            )
        elif installed_hnn_version != latest_stable:
            print(
                "\nWarning: you are attempting to build the textbook on a "
                "version of hnn-core that does not match the latest stable version."
                f"\n   Stable version: {latest_stable}"
                f"\n   Installed version: {installed_hnn_version}"
                "\n\nIf your installed version is behind the latest stable "
                "version, pase consider updating your local install before "
                "pushing any changes."
                "\n\nIf your installed version references a particular commit or "
                "branch (e.g.: hnn-core @ git+https://github.com/jonescompneurolab"
                "/hnn-core.git@1413550b2c610b700b7bb12ce7e1ae408ef8d4d3),"
                " we recommend that you use the --build-on-dev flag to specify "
                "the version of hnn-core that should be used."
            )

    convert_notebooks_to_html(
        input_folder=content_path,
        hash_path=hash_path,
        write_standalone_html=True,
        execute_notebooks=args.execute_notebooks,
        force_execute_all=args.force_execute_all,
        dev_build=commit_hash,
    )

    md_paths = get_page_paths()

    generate_page_html(
        md_paths,
        dev_build=args.build_on_dev,
    )


if __name__ == "__main__":
    main()
