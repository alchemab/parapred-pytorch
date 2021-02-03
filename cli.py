import click
import torch
import os
import json
import logging
import sys
from pprint import pprint
from typing import Optional

from parapred.model import Parapred, clean_output
from parapred.cnn import generate_mask
from parapred.preprocessing import encode_batch

MAX_PARAPRED_LEN = 40


LOGGER = logging.getLogger("Parapred-Logger")
LOGGER.setLevel(logging.INFO)

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parapred/weights/parapred_pytorch.h5")


@click.command(help = "Predict paratope probability for a given CDR sequence.")
@click.argument("cdr")
@click.option("--weight", "-w", help = "Specify path for weights.")
@click.option("--output", "-o", help = "Specify output JSON filename for prediction.", default = "output.json")
@click.option("--no-output", "-no", help = "Do not write an output file.", default = False, is_flag=True)
@click.option("--verbose", "-v", help = "Be verbose.", default = False, is_flag=True)
@click.option("--sigmoid", "-s", help = "Use sigmoid activation.", default = False, is_flag=True)
def predict(cdr: str,
            weight: Optional[str] = None,
            output: str = "output.json",
            no_output: bool = False,
            verbose: bool = False,
            sigmoid: bool = False):

    if len(cdr) > MAX_PARAPRED_LEN:
        LOGGER.error(f"Length of the CDR sequence ({len(cdr)}) is too long. Unsupported.")
        sys.exit(1)
    elif len(cdr) < 4:
        LOGGER.error(f"The original Parapred method requires at least 2 amino acids flanking the CDR.")
        sys.exit(1)

    # Encode input sequences
    if verbose:
        LOGGER.info(f"Encoding CDR sequence {cdr}")

    sequences, lengths = encode_batch([cdr], max_length=MAX_PARAPRED_LEN)

    # Generate a mask for the input
    m = generate_mask(sequences, sequence_lengths=lengths)

    # load pre-trained parapred model
    activation = "sigmoid" if sigmoid else "hard_sigmoid"
    if verbose and sigmoid:
        LOGGER.info(f"Using sigmoid activation in the LSTM.")

    p = Parapred(lstm_activation=activation)

    # load weights
    if verbose and weight is not None:
        LOGGER.info(f"Loading weights from {weight}")
        try:
            p.load_state_dict(torch.load(weight))
        except IOError:
            LOGGER.warning(f"Pre-trained weights file {weight} cannot be detected. Defaulting to pre-trained weights.")
            p.load_state_dict(torch.load(weight))

    elif verbose and weight is None:
        LOGGER.info(f"Loading pre-trained weights.")
        p.load_state_dict(torch.load(WEIGHTS_PATH))

    # Evaluation mode with no gradient computations
    _ = p.eval()
    with torch.no_grad():
        probabilities = p(sequences, m, lengths)

    # Linearise probabilities for viewing
    out = {}
    clean = clean_output(probabilities, lengths[0]).tolist()

    i_prob = [round(_, 5) for i, _ in enumerate(clean)]
    seq_to_prob = list(zip(cdr, i_prob))
    out[cdr] = seq_to_prob

    if verbose:
        pprint(out)

    if no_output is False:
        if verbose:
            LOGGER.info(f"Writing results to {output}")
        with open(output, "w") as jason:
            json.dump(out, jason)


@click.group()
def cli():
    pass

cli.add_command(predict)


if __name__ == "__main__":
    cli()
