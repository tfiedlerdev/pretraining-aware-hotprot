import argparse
from torch.utils.data import Dataset
import torch
import time
import os
import csv
from datetime import datetime
from typing import Optional
from util.telegram import TelegramBot
from util.esm import ESMEmbeddings
from util.prot_t5 import ProtT5Embeddings


class SequencesDataset(Dataset):
    def __init__(self, sequences: "set[str]") -> None:
        super().__init__()
        self.sequences = list(sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]


def generate_representations(
    dir_path,
    sequences,
    store_only_mean: bool,
    batch_size: int,
    model: str,
    telegram_bot: Optional[TelegramBot],
):
    if telegram_bot:
        telegram_bot.send_telegram(
            f"Generating remaining s_s representations for {len(sequences)} sequences"
        )
        response = telegram_bot.send_telegram(f"Generating first batch...")
    messageId = response["result"]["message_id"]

    os.makedirs(dir_path, exist_ok=True)

    ds = SequencesDataset(sequences)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=4
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
            embedding_generator = (
                ESMEmbeddings() if model == "esm" else ProtT5Embeddings()
            )
            emb = embedding_generator(sequences=inputs)

            batchesPredicted += 1
            with open(labels_file, "a") as csv:
                for s, data in enumerate(emb):
                    maxFilePrefix += 1
                    file_name = str(maxFilePrefix) + ".pt"
                    file_path = os.path.join(dir_path, file_name)
                    if not os.path.exists(file_path):
                        if store_only_mean:
                            data = data.mean(0)
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
                    f"Done with {index}/{numBatches} batches (hours to go: {int(hoursToGo)}) [last update: {now.hour}:{now.minute}]",
                )

            print(
                f"Done with {index}/{numBatches} batches (hours to go: {int(hoursToGo)}) [last update: {now.hour}:{now.minute}]"
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
    parser.add_argument("--store_only_mean", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--model", type=str, default="esm", choices=["esm", "protT5"])
    parser.add_argument("--telegram", action="store_true", default=False)

    args = vars(parser.parse_args())

    telegram_bot = TelegramBot() if args["telegram"] else None

    with open(args["input_file"], "r") as f:
        seqs = [line.replace("\n", "") for line in f.readlines()]

    print(f"Representations of {len(seqs)} sequences to be created")
    already_created_seqs = get_already_created_sequences(args["output_dir"])
    print(
        f"Representations of {len(already_created_seqs)} sequences of those already created"
    )
    remaining_seqs = set(seqs).difference(already_created_seqs)

    print(f"Creating remaining representations of {len(remaining_seqs)} sequences")

    try:
        generate_representations(
            args["output_dir"],
            remaining_seqs,
            args["store_only_mean"],
            batch_size=args["batch_size"],
            model=args["model"],
            telegram_bot=telegram_bot,
        )
        if telegram_bot:
            telegram_bot.send_telegram("Done!")
    except Exception as e:
        print("Exception raised: ", e)
        if telegram_bot:
            telegram_bot.send_telegram(
                "Generation of representations failed with error message: " + str(e)
            )
