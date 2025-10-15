# %%
import base64
import hashlib
import html
import json
import os
from pathlib import Path
import re
import textwrap
import warnings

import nbformat
import pypandoc
from hnn_core import __version__ as hnn_version
from nbconvert.preprocessors import (
    ClearOutputPreprocessor,
    ExecutePreprocessor,
)
from packaging.version import Version

textbook_root_path = Path(__file__).parents[1]

def _save_plot_as_image(
    img_data,
    img_filename,
    output_dir,
):
    """Saves the plot image to the specified directory."""
    img_path = os.path.join(
        output_dir,
        img_filename,
    )
    with open(img_path, "wb") as img_file:
        img_file.write(base64.b64decode(img_data))
    return


def _html_to_json(
    html: str,
    filename: str,
):
    """
    Convert html into hierarchical json
    """
    # variable for processed json output
    contents = {filename: {}}

    # variable to track section content and metadata
    current_html = list()
    current_title = None
    current_level = None

    # split html into lines while replacing tabs with spaces
    lines = [
        line.replace(
            "\t",
            "    ",
        )
        for line in html.splitlines()
        # if line.strip()
    ]

    for i, line in enumerate(lines):
        # identify lines with header tag
        # note: the match is performed on the line stripped of any
        # spaces or newlines
        line_match = re.match(
            r"(<h[1-6]>)(.*?)(</h[1-6]>)",
            line.strip(),
        )

        if line_match:
            # when a new header is found, save the previous section
            if current_title:
                contents[filename][current_title] = {}
                contents[filename][current_title]["level"] = current_level
                contents[filename][current_title]["html"] = "\n".join(current_html)

            # get the title, level of the new section
            current_level = line_match.group(1).strip()
            current_level = int(current_level.lstrip("<h").rstrip(">"))
            current_title = line_match.group(2).strip()

            # start a new section with the previous line
            current_html = [lines[i - 1]]

        elif current_html is not None:
            # add new html lines
            current_html.append(lines[i - 1])

    # save the last section
    if current_title:
        # append the last line
        current_html.append(line)

        # update contants
        contents[filename][current_title] = {}
        contents[filename][current_title]["level"] = current_level
        contents[filename][current_title]["html"] = "\n".join(current_html)

    return contents


def _structure_json(contents):
    """
    Determine the hierarchy of sections based on levels without adding content.
    Returns a list of sections in order of their hierarchy.
    """
    hierarchy = {}

    for filename, sections in contents.items():
        hierarchy[filename] = {}

        # list to track parent sections for potential nesting
        parent_stack = []

        for section_title, section_data in sections.items():
            level = section_data["level"]
            html_contents = section_data["html"]

            # Create a section dict with 'title', 'level', and 'sub-sections'
            section_info = {
                "title": section_title,
                "level": level,
                "html": html_contents,
                "sub-sections": [],
            }

            # Ensure only sections with a level greater than the current
            # section remain in the stack as potential parents
            while parent_stack and parent_stack[-1]["level"] >= level:
                parent_stack.pop()

            if parent_stack:
                # Add the section as a child of the last parent
                parent_stack[-1]["sub-sections"].append(section_info)
            else:
                # Add the section as a top-level section
                hierarchy[filename][section_title] = section_info

            # Add the current section to the parent stack for future nesting
            parent_stack.append(section_info)

    def remove_blank_subsections(sections):
        seek = "sub-sections"

        for k, v in list(sections.items()):
            if isinstance(v, dict):
                # check for 'sub-sections' key in dict
                if seek in v:
                    # delete empty sub-sections
                    if v[seek] == []:
                        del v[seek]

                    # Recursively check all sub-sections
                    else:
                        for sub_section in v[seek]:
                            remove_blank_subsections(sub_section)

            elif isinstance(v, list):
                # if v is an empty list, delete it
                if v == []:
                    del sections[k]
                # if v is a list of dicts, iterate through the dicts
                else:
                    for dictionary in v:
                        remove_blank_subsections(dictionary)

        return sections

    hierarchy[filename] = remove_blank_subsections(hierarchy[filename])

    return hierarchy


def _extract_html_from_nb(
    nb,
    input_dir,
    filename,
    dev_build=False,
    use_base64=False,
):
    """Extracts HTML for cell contents and outputs,
    including code and markdown."""

    html_output = []
    fig_id = 0
    delim = os.path.sep
    aggregated_output = ""

    # helper for aggregating outputs
    # -----------------------------
    def _aggregate_outputs(
        html_output,
        accumulated_outputs,
    ):
        """
        If there are accumulated outputs, append them to html_output and reset.
        """
        if accumulated_outputs:
            cell_output_html = textwrap.dedent(f"""
                <!-- code cell output -->
                <div class='output-cell'>
                    <div class='output-label'>
                        Out:
                    </div>
                    <div class='output-code'>
                        {accumulated_outputs}
                    </div>
                </div>
            """)
            html_output.append(cell_output_html)
        return ""

    for cell in nb["cells"]:
        # ------------------------------
        # process code cells
        # ------------------------------
        if cell["cell_type"] == "code":
            # ==============================
            # add code cell contents
            # ==============================
            code_cell_html = textwrap.dedent(f"""
                <!-- code cell -->
                <div class='code-cell'>
                    <code class='language-python'>
                        {cell["source"]}
                    </code>
                </div>
            """)
            html_output.append(code_cell_html)

            # ==============================
            # add code cell outputs
            # ==============================
            for output in cell.get("outputs", []):
                # handle plain text outputs
                # ------------------------------
                if "text/plain" in output.get("data", {}):
                    text_output = output["data"]["text/plain"]
                    # escape '<' and '>' characters which can be
                    # incorrectly interpreted as HTML tags
                    escaped_text_output = html.escape(text_output)

                    # aggregate outputs
                    aggregated_output += f"\n\t\t{escaped_text_output}"

                # handle stdout
                # ------------------------------
                # e.g., this includes outputs from print statements
                if (
                    output.get("output_type") == "stream"
                    and output.get("name") == "stdout"
                ):
                    stream_output = output.get("text", "")
                    # escape '<' and '>' characters
                    escaped_stream_output = html.escape(stream_output)

                    # aggregate outputs
                    aggregated_output += f"\n\t\t{escaped_stream_output}"

                # handle image outputs (e.g., plots)
                # ------------------------------
                if "image/png" in output.get("data", {}):
                    # before processing the image, append any already-accumulated
                    # outputs to html_output and re-set aggregated_output to ""
                    aggregated_output = _aggregate_outputs(
                        html_output,
                        aggregated_output,
                    )

                    img_data = output["data"]["image/png"]

                    # optional Base64 encoding for image embedding
                    # ------------------------------
                    if use_base64:
                        output_base64_html = textwrap.dedent(f"""
                            <!-- code cell image -->
                            <div class='output-cell'>
                                <img src='data:image/png;base64,{img_data}'/>
                            </div>
                        """)
                        html_output.append(output_base64_html)

                    # standard image processing using saved .png file
                    # ------------------------------
                    else:
                        fig_id += 1
                        img_filename = f"fig_{fig_id:02d}.png"

                        output_folder = "output_nb_" + f"{filename.split('.ipynb')[0]}"
                        output_dir = f"{input_dir}{delim}{output_folder}"

                        # if doing a dev build, switch the output directory
                        if dev_build:
                            output_dir = output_dir.replace(
                                "content",
                                "dev",
                            )

                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)

                        _save_plot_as_image(
                            img_data,
                            img_filename,
                            output_dir,
                        )

                        output_img_html = textwrap.dedent(f"""
                            <!-- code cell image -->
                            <div class='output-cell'>
                                <img src='{output_folder}{delim}{img_filename}'/>
                            </div>
                        """)
                        html_output.append(output_img_html)

                # handle errors
                # ------------------------------
                if output.get("output_type") == "error":
                    error_message = "\n".join(
                        output.get(
                            "traceback",
                            [],
                        ),
                    )
                    output_error_html = textwrap.dedent(f"""
                        <!-- code cell error -->
                        <div class='output-cell error'>
                            <pre>
                                {error_message}
                            </pre>
                        </div>
                    """)
                    html_output.append(output_error_html)

            # ==============================
            # accumulate remaining cell outputs
            # ==============================
            # If there are accumulated outputs for the cell that have not
            # yet been added to the html, append the outputs to html_output
            aggregated_output = _aggregate_outputs(
                html_output,
                aggregated_output,
            )

        # ------------------------------
        # process "markdown" cells
        # ------------------------------
        elif cell["cell_type"] == "markdown":
            # escape < and > characters
            markdown_content = html.escape(cell["source"])

            html_content = pypandoc.convert_text(
                markdown_content,
                format="md",
                to="html",
                extra_args=[
                    "--mathml",
                    # the "-f", "markdown-auto_identifiers" arguments below
                    # disable the automatic ids added to header tags
                    "-f",
                    "markdown-auto_identifiers",
                ],
            )
            markdown_html_output = textwrap.dedent(f"""
                <!-- markdown cell -->
                <div class='markdown-cell'>
                    {html_content}
                </div>
            """)
            html_output.append(markdown_html_output)

    html_output = "\n".join(html_output)

    return html_output


def _hash_nb(nb_path):
    """Generate a SHA256 hash of the notebook, ignoring outputs/metadata."""

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # clear all cell outputs
    preprocessor = ClearOutputPreprocessor()
    preprocessor.preprocess(nb, {})

    # remove execution counts and cell metadata
    for cell in nb.cells:
        if "execution_count" in cell:
            cell["execution_count"] = None
        if "metadata" in cell:
            cell["metadata"] = {}

    # remove nb metadata
    nb.metadata = {}

    # serialize cleaned nb
    nb_json = nbformat.writes(nb, version=4).encode("utf-8")

    # generate hash
    hasher = hashlib.sha256()
    hasher.update(nb_json)

    return hasher.hexdigest()


def _load_nb_hashes(nb_hash_path):
    """Load previously-recorded hashes notebook hashes"""
    # AES if we want to support optional or fresh hash building, we should do it at the
    # CLI in main, not here, but leaving this as-is for now.
    if os.path.exists(nb_hash_path):
        with open(nb_hash_path, "r") as f:
            return json.load(f)
    return {}


def _save_nb_hashes(
    new_hashes,
    nb_hash_path,
):
    """Save updated notebook hashes"""

    # print(f'Saving hashes to {nb_hash_path}')
    with open(nb_hash_path, "w") as f:
        json.dump(new_hashes, f, indent=4)


def _load_nb(nb_path):
    """Get a jupyter notebook object and optionally execute it"""
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    return nb


def _is_nb_fully_executed(nb):
    """
    Check if a notebook object has been fully executed.

    Returns True if all code cells have an associated execution_count, excluding those
    with no source code.
    """
    for cell in nb.get("cells", []):
        if (
            (cell.get("cell_type") == "code")
            and (cell.get("execution_count") is None)
            and (cell.get("source") != "")
        ):
            return False

    return True


def _nb_has_json_output(
    nb_path,
    output_dir,
):
    """
    Check if the notebook has been fully executed by checking against the
    json output file.

    AES TODO write out the return types since they're heterogeneous
    """

    json_path = output_dir / f"{nb_path.stem}.json"

    execution_check = False
    version_check = False
    commit_check = False

    # if the json output exists, get the execution status, base version,
    # and latest commit used to execute the nb
    if json_path.exists():
        with open(json_path, "r") as file:
            nb_outputs = json.load(file)
            execution_check = nb_outputs.get(
                "full_executed",
                False,
            )
            version_check = nb_outputs.get(
                "hnn_version",
                False,
            )
            commit_check = nb_outputs.get(
                "commit",
                False,
            )

    return commit_check, execution_check, version_check


def _setup_root_and_input(input_folder):
    """
    if the current directory doesn't end in 'textbook', recursively look
    for 'textbook' in the parent directories
    """
    root = os.getcwd()
    while os.path.basename(root) != "textbook":
        root = os.path.dirname(root)

    # default input folder is the 'content' folder, which is
    # the directory from which our site is published
    if not input_folder:
        input_folder = os.path.join(root, "content")

    return root, input_folder


def _load_nbs_to_skip(nb_skip_path, dev_build):
    """
    Get the list of notebooks to skip from the 'nbs_to_skip.json' file in the 'nb_skip_path'.

    The "dev_build" flag determines which list is extracted from the json.
    """
    with open(nb_skip_path, "r") as f:
        nbs_to_skip = json.load(f)

    if dev_build:
        # AES appears to be "skip if dev"
        # AES maybe most are in the "skip if dev" for debugging? not sure why
        nbs_to_skip = nbs_to_skip["skip_if_dev"]
    else:
        # AES appears to be "skip if stable"
        nbs_to_skip = nbs_to_skip["skip_if_stable"]

    return nbs_to_skip


def _execute_nb(nb_path, timeout=600):
    execution_initiated = True
    print(f"Execution: Notebook {nb_path.name} execution has been initiated.")

    loaded_nb = _load_nb(nb_path)

    ep = ExecutePreprocessor(
        timeout=timeout,
        kernel_name="python3",
    )
    ep.preprocess(
        loaded_nb,
        {"metadata": {"path": os.path.dirname(nb_path)}},
    )

    execution_successful = _is_nb_fully_executed(
        loaded_nb,
    )
    if execution_successful:
        print(f"Execution: Notebook {nb_path.name} execution has been successful.")

    return loaded_nb, execution_initiated, execution_successful


def _process_nb(
    root,
    nb_path,
    filename,
    current_directory,
    nb_hashes,
    nbs_to_skip,
    dev_build,
    output_dir,
    execution_filter,
):
    """
    Execute notebooks as needed and return the updated hash and execution contents.

    AES: this docstring also included "convert [nbs] to html and json", but as far as I
    can tell, this function no longer does that, so I have removed it.
    """

    # get the nb without executing it
    loaded_nb = _load_nb(nb_path)

    # hash the nb in its current state
    current_hash = _hash_nb(nb_path)

    # flag for whether the nb was run, initialized as false
    current_execution_initiated = False

    # Check if the nb has been fully executed, and get the nb_version as well as the
    # commit hash. We will use this in a warning in the skipped case, or actually use
    # this data in other cases.
    prior_commit_if_any, prior_execution_if_any, prior_version_if_any = (
        _nb_has_json_output(
            nb_path=nb_path,
            output_dir=output_dir,
        )
    )

    # determine if nb should be executed
    print(f"Configuration: Checking whether '{filename}' should be newly re-executed.")
    should_execute = _determine_should_execute_nb(
        nb_hashes,
        current_hash,
        dev_build,
        prior_commit_if_any,
        prior_execution_if_any,
        prior_version_if_any,
        nbs_to_skip=nbs_to_skip,
        nb_path=nb_path,
        execution_filter=execution_filter,
    )
    # execute nb as needed
    if should_execute:
        loaded_nb, current_execution_initiated, current_execution_successful = (
            _execute_nb(nb_path)
        )
        execution_successful = current_execution_successful

        # AES: I've tried to reorganize the warnings. The warnings here are now only for
        # issues with a current execution attempt. All warnings related to prior
        # execution attempts or skipping are inside
        # _determine_should_execute_nb. Original comment follows:
        # --------------------------------
        # warning for the case when nb execution was attempted
        # but the nb was not fully executed for some reason
        if not current_execution_initiated:
            # AES This is a new warning.
            warnings.warn(
                "\n\n"
                "# -------------------------------------------------------\n"
                f"# ERROR: Execution of notebook '{filename}'"
                "# could not be initiated\n"
                "# successfully. Please investigate the notebook\n"
                "# to determine why execution was not successfully initiated.\n"
                "# The html and json outputs may be incomplete."
                "# -------------------------------------------------------"
                "\n\n"
            )
        elif (current_execution_initiated) and (not current_execution_successful):
            warnings.warn(
                "\n\n"
                "# -------------------------------------------------------\n"
                f"# ERROR: Execution of notebook '{filename}'"
                "# was initiated but did not complete\n"
                "# successfully. Please investigate the notebook\n"
                "# to determine why execution was not successfully completed.\n"
                "# The html and json outputs may be incomplete."
                "# -------------------------------------------------------"
                "\n\n"
            )
    else:
        execution_successful = prior_execution_if_any

    return current_hash, loaded_nb, current_execution_initiated, execution_successful


def _determine_should_execute_nb(
    nb_hashes,
    current_hash,
    dev_build,
    prior_commit_if_any,
    prior_execution_if_any,
    prior_version_if_any,
    nbs_to_skip,
    nb_path,
    execution_filter,
):
    """
    TODO Determine whether or not a notebook should be executed based on
    various factors, including:
        - Has the notebook been executed previously
        - Is the notebook "new" (i.e., it is not associated with a json output)
        - Is the user performing a 'dev' build
        - Has the notebook hash changed since the last execution

    A warning will be printed to the terminal if execute_notebooks is False
    and the locally-installed version of hnn-core is greater than the version
    last used to run the notebook

    Inputs
    ------
    filename : str
    nb_hashes : dict
        Mapping of notebook filenames to their previously-determined hash values,
        loaded from notebook_hashes.json
    current_hash : str
        Newly-determined notebook hash based on the file's current state
    dev_build : str or bool
        False if not running a dev build. Otherwise, this variable will be
        a string containing the repo and commit hash to be used for the build
    prior_commit_if_any : str
        Contains the commit hash from the previous execution, loaded from the
        notebook's corresponding json output file. Used for checking/validating
        versions when doing a 'dev' build
    prior_execution_if_any : bool
        Flag for whether or not the notebook is already executed per the
        notebook's corresponding json output file
    prior_version_if_any : str
        The version of hnn-core that was last used to execute the notebook

    Returns
    -------
        bool : a boolean indicating if the notebook should be executed
    """

    filename = nb_path.name

    # 1) handle super omega execute all notebooks, including skipped
    #     - This is a brand-new option.
    if execution_filter == "execute-absolutely-all-notebooks":
        print(
            "Execution set to all notebooks, including skipped! CHARGE PROTON TORPEDOS!"
        )
        return True

    # 2) handle no execution of any notebooks
    #     - This was formerly the "silent default" behavior of "python build.py" with no
    #     args.
    if execution_filter == "no-execution":
        # 2.1) if nb new
        if filename not in nb_hashes:
            print(
                f"Notebook {filename} appears to be new and needs to be executed."
                f"Not performing execution since execution_filter is set to '{execution_filter}'.\n"
            )
        # 2.2) if nb hash has changed
        elif nb_hashes[filename] != current_hash:
            print(
                f"Notebook {filename} appears to have been updated and needs to be executed."
                f"Not performing execution since execution_filter is set to '{execution_filter}'.\n"
            )
        # 2.3) warning if version out of date
        elif prior_version_if_any != "NA":
            if Version(hnn_version) > Version(prior_version_if_any):
                warnings.warn(
                    "\n\n"
                    "# -------------------------------------------------------\n"
                    f"# WARNING: Notebook {filename} may have been executed on an\n"
                    "# older version of hnn-core, as your installed version\n"
                    "# is greater than version used to run the notebook\n"
                    "# previously. Please consider re-executing this notebook."
                    "\n#\n"
                    "# Last version used to run notebook:\n"
                    f"#    {prior_version_if_any}\n"
                    "# Installed version:\n"
                    f"#    {hnn_version}\n\n"
                    f"# Not performing execution since execution_filter is set to '{execution_filter}'.\n"
                    "# -------------------------------------------------------"
                    "\n\n"
                )
        # 2.4) warning if prior execution not successful
        elif not prior_execution_if_any:
            warnings.warn(
                "\n\n"
                "# -------------------------------------------------------\n"
                f"# WARNING: Notebook '{filename}' does not\n"
                "# appear to have been successfully\n"
                "# executed the last time it was run.\n"
                "# The html and json output may be incomplete.\n"
                f"# Not performing execution since execution_filter is set to '{execution_filter}'.\n"
                "# -------------------------------------------------------"
                "\n\n"
            )

        return False

    # 3) In all other cases (see below), skip first if possible
    skip_nb = filename in nbs_to_skip
    if skip_nb:
        print(
            f"Notebook '{filename}' has been flagged to be"
            " skipped. Execution will not be attempted for"
            " this notebook."
        )
        if not prior_execution_if_any:
            warnings.warn(
                "\n\n"
                "# -------------------------------------------------------\n"
                f"# WARNING: '{filename}' is flagged to be skipped, but does not\n"
                "# appear to have been successfully\n"
                "# executed the last time it was run.\n"
                "# The html and json output may be incomplete.\n"
                "\n#\n"
                "# Please either remove the notebook from the skipped list JSON file,\n"
                "# or re-run the script with\n"
                "# '--execution-filter=execute-absolutely-all-notebooks'\n"
                "# to ensure that the notebook outputs are correct.\n"
                "# -------------------------------------------------------"
                "\n\n"
            )
        elif prior_version_if_any == "NA":
            warnings.warn(
                "\n\n"
                "# -------------------------------------------------------\n"
                f"# WARNING: '{filename}' is flagged to be skipped, but does not\n"
                "# appear to have been previously executed ever.\n"
                "\n#\n"
                "# Please either remove the notebook from the skipped list JSON file,\n"
                "# or re-run the script with\n"
                "# '--execution-filter=execute-absolutely-all-notebooks'\n"
                "# to ensure that the notebook outputs are correct.\n"
                "# -------------------------------------------------------"
                "\n\n"
            )

        return False

    # 4) Handle executing everything except skipped, since skippable nbs have already
    # been skipped above.
    #     - This was formerly called via the "--force-execute-all" CLI arg.
    if execution_filter == "execute-all-unskipped-notebooks":
        print(f"Executing {filename}")
        return True

    # 5) handle "regular" execute
    #     - This was formerly called via the "--execute-notebooks" CLI arg.
    if execution_filter == "execute-updated-unskipped-notebooks":
        # --------------------------------------------------
        # 4.1) if the hash has not changed
        if (filename in nb_hashes) and (nb_hashes[filename] == current_hash):
            if dev_build:
                # check if the commit specified to use by the dev build
                # matches the commit last used to run the nb per the
                # prior_commit_if_any (returned by _nb_has_json_output)
                #
                # if the versions do not match, the nb is flagged to
                # be re-executed by setting "prior_execution_if_any=False"
                if dev_build != prior_commit_if_any:
                    prior_execution_if_any = False
                    print(f"Executing {filename} due to dev build commit mismatch.")
                    return True
            elif not prior_execution_if_any:
                print(f"Executing {filename} due to previous failure of execution.")
                return True
            else:
                print(
                    f"Not executing: Notebook {filename} is unchanged and already fully executed."
                )
                return False
        else:
            print(f"Executing {filename} due to notebook being either new or changed.")
            return True


def _write_standalone_nb_to_html(
    html_content,
    current_directory,
    filename,
    dev_build=False,
):
    """
    Write a notebook to a standalone HTML file. Note that the standalone html
    files are not part of the textbook website. They merely provide a "snapshot" of
    how the notebook is rendered in html. Writing standalone html files can be useful
    to check how unpublished notebooks or notebooks in development will look on the
    website once they are publishes

    Inputs
    ------
    html_content : str
        The html content extracted from the notebook
    current_directory: str
        The current working directory
    filename : str
        The name of the file being processed
    dev_build : str or bool
        False if not running a dev build. Otherwise, this variable will be
        a string containing the repo and commit hash to be used for the build

    """

    if dev_build:
        dev_dir = current_directory.replace(
            "content",
            "dev",
        )
        output_file = os.path.join(
            dev_dir,
            f"{os.path.splitext(filename)[0]}.html",
        )
        if not os.path.exists(dev_dir):
            os.makedirs(dev_dir)
    else:
        output_file = os.path.join(
            current_directory,
            f"{os.path.splitext(filename)[0]}.html",
        )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("<html><body>\n")
        f.write(html_content)
        f.write("\n</body></html>")

    return


def _write_nb_json(
    html_content,
    filename,
    current_directory,
    execution_initiated,
    execution_successful,
    dev_build=False,
):
    """
    Generate structured json output for the notebook.
    """

    # ----------------------------------------
    # generated structured json output
    # ----------------------------------------
    # Note: this section pertains to a planned enhancement
    # to enable inserting sections of a nb into an
    # html file by specifing the headers to include; e.g.,
    # including [[notebook][start header][end header]] in your
    # .md file would inject only the .html for those header
    # sections into your html output file

    nb_html_json = _html_to_json(
        html_content,
        filename,
    )

    if dev_build:
        output_json = os.path.join(
            current_directory.replace("content", "dev"),
            f"{os.path.splitext(filename)[0]}.json",
        )
        os.makedirs(
            current_directory.replace("content", "dev"),
            exist_ok=True,
        )
    else:
        output_json = os.path.join(
            current_directory,
            f"{os.path.splitext(filename)[0]}.json",
        )

    # AES why not just use nb_exec?
    if execution_initiated:
        # Add execution status directly to json output
        # Track version used in nb execution
        # AES if nb was started, but failed, then full_executed would be false here
        # AES maybe the point is "this is the last version they were successfully executed with"
        nb_html_json = {
            "full_executed": execution_successful,
            "hnn_version": hnn_version,
            **nb_html_json,
        }
        if dev_build:
            print("Dev version to use:", dev_build)
            nb_html_json["commit"] = dev_build
    else:
        # get previously-used hnn version from json file
        previous_version = "NA"
        if os.path.exists(output_json):
            with open(output_json, "r") as f:
                nb_html_json = json.load(f)
            # check for hnn_version key
            if "hnn_version" in nb_html_json:
                previous_version = nb_html_json["hnn_version"]
        nb_html_json = {
            "full_executed": execution_successful,
            "hnn_version": previous_version,
            **nb_html_json,
        }
        if dev_build:
            nb_html_json["commit"] = dev_build

    with open(output_json, "w") as f:
        json.dump(nb_html_json, f, indent=4)

    return output_json


# AES I think most of these arguments should be required, not optional
def execute_and_convert_nbs_to_json(
    input_folder=None,
    use_base64=False,
    write_standalone_html=False,
    dev_build=False,
    nb_hash_path=Path(textbook_root_path / "scripts" / "nb_hashes.json"),
    nb_skip_path=Path(textbook_root_path / "scripts" / "nbs_to_skip.json"),
    execution_filter=None,
):
    """
    Executes and converts .ipynb files in the input folder to JSON (and optionally HTML).
    """

    # ==================== #
    #        SETUP
    # ==================== #
    root, input_folder = _setup_root_and_input(input_folder)

    # get nb hashes from json
    nb_hashes = _load_nb_hashes(nb_hash_path)
    updated_hashes = nb_hashes.copy()

    # get list of nbs to skip
    nbs_to_skip = _load_nbs_to_skip(nb_skip_path, dev_build)

    # ==================== #
    # Loop through notebooks
    # ==================== #

    # iterate through input directory and process notebooks
    for current_directory, _, list_files in os.walk(input_folder):
        for filename in list_files:
            if not filename.endswith(".ipynb"):
                continue

            print(f"\nProcessing notebook: {filename}")

            # get the path to the nb
            nb_path = os.path.join(current_directory, filename)


            # AES TODO as refactor with pathlib, expand Path usage
            nb_path = Path(nb_path)
            if dev_build:
                # Replace "content" parent directory with "dev" one
                output_dir = Path(str(nb_path).replace("content", "dev"))
                output_dir = output_dir.parents[0]
                output_dir.mkdir(parents=True, exist_ok=True)
            else:
                output_dir = nb_path.parents[0]

            # process nb and update hash
            processed_hash, loaded_nb, execution_initiated, execution_successful = (
                _process_nb(
                    root=root,
                    nb_path=nb_path,
                    filename=filename,
                    current_directory=current_directory,
                    nb_hashes=nb_hashes,
                    nbs_to_skip=nbs_to_skip,
                    dev_build=dev_build,
                    output_dir=output_dir,
                    execution_filter=execution_filter,
                )
            )

            # extract and process the html from the nb
            html_content = _extract_html_from_nb(
                loaded_nb,
                current_directory,
                filename,
                dev_build=dev_build,
                use_base64=use_base64,
            )

            # generate complete json output file
            _write_nb_json(
                html_content,
                filename,
                current_directory,
                execution_initiated,
                execution_successful,
                dev_build=dev_build,
            )

            # optionally write standalone nb to an html file
            # note: standalone nbs are NOT part of the website build; rather,
            #   they offer a "snapshot" view of how the nb will look when
            #   rendered in html. This is useful when creating/testing new nbs
            if write_standalone_html:
                html_content = _write_standalone_nb_to_html(
                    html_content,
                    current_directory,
                    filename,
                    dev_build,
                )

            print(f"Successfully converted '{filename}' to html")

            updated_hashes[filename] = processed_hash

    # save updated hashes
    _save_nb_hashes(
        updated_hashes,
        nb_hash_path,
    )

    return


# # AES TODO
# # %%

# run_test = False


# def test_nb_conversion(input_folder=None):
#     execute_and_convert_nbs_to_json(
#         input_folder=input_folder,
#         use_base64=False,
#         write_standalone_html=True,
#         execute_nbs=True,
#     )


# if run_test:
#     test_nb_conversion("tests")
