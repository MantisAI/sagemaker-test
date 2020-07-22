import json
import logging

logger = logging.getLogger(__name__)


def write_jsonl(input_data: list, output_file: str, mode: str = "w"):
    """
    Write a dict to jsonl (line delimited json)

    Output format will look like:

    ```
    {'a': 0}
    {'b': 1}
    {'c': 2}
    {'d': 3}
    ```

    Args:
        input_data(dict): A dict to be written to json.
        output_file(str): Filename to which the jsonl will be saved.
    """

    try:
        with open(output_file, mode) as fb:

            # Check if a dict (and convert to list if so)

            if isinstance(input_data, dict):
                input_data = [value for key, value in input_data.items()]

            # Write out to jsonl file

            logger.debug("Writing %s lines to %s", len(input_data), output_file)

            for i in input_data:
                json_ = json.dumps(i) + "\n"
                fb.write(json_)

        logger.debug(f"Wrote {len(input_data)} data to {output_file}")

    except Exception:
        logger.exception(f"Problem writing to {output_file}")


def _yield_jsonl(file_name):
    for row in open(file_name, "r"):
        yield json.loads(row)


def read_jsonl(input_file):
    """Create a list from a jsonl file

    Args:
        input_file(str): File to be loaded.
    """

    try:
        out = list(_yield_jsonl(input_file))
        logger.debug(f"Read {len(out)} lines from {input_file}")
        return out
    except Exception:
        logger.exception(f"Problem reading {input_file}")
