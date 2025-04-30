import esm
from esm import FastaBatchedDataset
import pathlib
import torch
from tqdm import tqdm


def esm2(ids, seqs, output_dir_name, model, alphabet):
    
    output_dir = pathlib.Path(output_dir_name)
    model.eval()
    tokens_per_batch = 4096
    seq_length = 1022
    repr_layers= [30]
    if torch.cuda.is_available():
        model = model.cuda()

    dataset = FastaBatchedDataset(ids, seqs)
    batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)
    results = []
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(seq_length),
        batch_sampler=batches
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in tqdm(enumerate(data_loader)):
              #print(f'Processing batch {batch_idx + 1} of {len(batches)}')
              if torch.cuda.is_available():
                  toks = toks.to(device="cuda", non_blocking=True) #cuda

              out = model(toks, repr_layers=repr_layers, return_contacts=False)

              logits = out["logits"].to(device="cpu")
              representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}

              for i, label in enumerate(labels):
                  entry_id = label.split()[0]

                  truncate_len = min(seq_length, len(strs[i]))

                  filename = output_dir / f"{entry_id}.pt"
                  result = {"entry_id": entry_id}
                  result["representations"] = {
                          layer: t[i, 1 : truncate_len + 1].clone()
                          for layer, t in representations.items()
                      }
                  results.append(result)

                  torch.save(result, filename)