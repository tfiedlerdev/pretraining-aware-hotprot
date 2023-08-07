import argparse
from torch.utils.data import Dataset
import torch
import time
import os
import csv
from datetime import datetime
from typing import Optional
from util.telegram import TelegramBot
from util.esmfold import ESMFoldEmbeddings
from util.prot_t5 import ProtT5Embeddings
from util.esm import ESMEmbeddings
import traceback
from thermostability.hotinfer import RepresentationKeysComb


class SequencesDataset(Dataset):
    def __init__(self, sequences: "set[str]", max_len: int = 700) -> None:
        super().__init__()
        self.sequences = [
            sequence for sequence in list(sequences) if len(sequence) <= max_len
        ]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]


def generate_representations(
    dir_path,
    sequences,
    batch_size: int,
    model: str,
    repr_key: RepresentationKeysComb,
    telegram_bot: Optional[TelegramBot],
):
    if telegram_bot:
        telegram_bot.send_telegram(
            f"Generating remaining s_s representations for {len(sequences)} sequences"
        )
        response = telegram_bot.send_telegram("Generating first batch...")
        messageId = response["result"]["message_id"]

    os.makedirs(dir_path, exist_ok=True)

    ds = SequencesDataset(sequences)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=1
    )
    timeStart = time.time()
    labels_file = os.path.join(dir_path, "sequences.csv")
    if not os.path.exists(labels_file):
        with open(labels_file, "w") as csv:
            csv.write("sequence, filename\n")

    file_prefixes = [
        int(fname.split(".")[0])
        for fname in os.listdir(dir_path)
        if fname != "sequences.csv"
    ]
    maxFilePrefix = max(file_prefixes) if len(file_prefixes) > 0 else 0
    print(f"Starting with maxFilePrefix {maxFilePrefix}")
    batchesPredicted = 0

    for index, (inputs) in enumerate(loader):
        batch_size = len(inputs)
        numBatches = int(len(sequences) / batch_size)
        print(f"At batch {index}/{numBatches}")
        with torch.no_grad():
            print("Predicting")
            if model == "esm":
                embedding_generator = ESMFoldEmbeddings()
            elif model == "protT5":
                embedding_generator = ProtT5Embeddings()
            else:
                embedding_generator = ESMEmbeddings(model)

            emb = embedding_generator(sequences=inputs, representation_key=repr_key)
            batchesPredicted += 1
            with open(labels_file, "a") as csv:
                for s, data in enumerate(emb):
                    maxFilePrefix += 1
                    file_name = str(maxFilePrefix) + ".pt"
                    file_path = os.path.join(dir_path, file_name)
                    if not os.path.exists(file_path):
                        with open(file_path, "wb") as f:
                            torch.save(data.cpu(), f)
                        csv.write(f"{inputs[s]}, {file_name}\n")
        if index % 5 == 0:
            secsSpent = time.time() - timeStart
            secsToGo = (secsSpent / (batchesPredicted + 1)) * (numBatches - index - 1)
            hoursToGo = secsToGo / (60 * 60)
            now = datetime.now()
            if telegram_bot:
                telegram_bot.edit_text_message(
                    messageId,
                    f"""Done with {index}/{numBatches} batches (hours to go:
                      {int(hoursToGo)}) [last update: {now.hour}:{now.minute}]""",
                )

            print(
                f"""Done with {index}/{numBatches} batches (hours to go: {int(hoursToGo)})
                 [last update: {now.hour}:{now.minute}]"""
            )


def get_already_created_sequences(dir_path):
    labels_file = os.path.join(dir_path, "sequences.csv")
    if os.path.exists(labels_file):
        with open(labels_file, "r") as f:
            return [
                seq
                for i, (seq, _) in enumerate(
                    csv.reader(f, delimiter=",", skipinitialspace=True)
                )
                if i != 0
            ]
    return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        type=str,
        help="File containing amino ascid sequences split by new line for which to generate representations",
    )
    parser.add_argument(
        "output_dir", type=str, help="Directory in which to place the representations"
    )
    parser.add_argument("model", type=str, default="esm")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--telegram", action="store_true", default=False)
    parser.add_argument(
        "--repr_key",
        type=str,
        default="s_s_avg",
        choices=[
            "prott5_avg",
            "prott5",
            "esm_s_A",
            "esm_s_B",
            "s_s_0_A",
            "s_s_0_B",
            "esm_s_B_avg",
            "s_s_avg",
            "esm_3B",
            "esm_650M",
            "esm_150M",
            "esm_35M",
            "esm_8M",
            "s_s"
        ],
    )
    parser.add_argument("--max_seq_len", type=int, default=700)

    args = vars(parser.parse_args())

    telegram_bot = TelegramBot() if args["telegram"] else None

    with open(args["input_file"], "r") as f:
        seqs = [line.replace("\n", "") for line in f.readlines()]

    print(f"Representations of {len(seqs)} sequences to be created")
    already_created_seqs = get_already_created_sequences(args["output_dir"])
    print(
        f"Representations of {len(already_created_seqs)} sequences of those already created"
    )
    remaining_seqs = [seq for seq in set(seqs).difference(already_created_seqs) if len(seq) < args["max_seq_len"]]

    print(f"Creating remaining representations of {len(remaining_seqs)} sequences")

    try:
        generate_representations(
            args["output_dir"],
            remaining_seqs,
            batch_size=args["batch_size"],
            repr_key=args["repr_key"],
            model=args["model"],
            telegram_bot=telegram_bot,
        )
        if telegram_bot:
            telegram_bot.send_telegram("Done!")
    except Exception as e:
        print("Exception raised: ", e)
        print(traceback.format_exc())
        if telegram_bot:
            telegram_bot.send_telegram(
                "Generation of representations failed with error message: " + str(e)
            )
