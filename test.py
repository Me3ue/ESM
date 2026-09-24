import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import EsmForProteinFolding
from transformers.models.esm.openfold_utils.protein import Protein, to_pdb
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

model = EsmForProteinFolding.from_pretrained(
    "facebook/esmfold_v1",
    torch_dtype=torch.bfloat16,
)
model = model.eval().cuda()
model.trunk.set_chunk_size(32)

sequence = (
    "LTPEEQELIEKARAVVESIPASEDHSVAAAALSADGRVFTGVNVYHPGGGPCAEQVVLG"
    "VAAAAAAGPLTAIVCYGQNGRGILSPCGKCRGILAALQPGVRAIVLDENGQPVAVPIAS"
    "LLPALSPEAKELITKATEVINSIPKSDVHNVAAAALAEDGSIYTGVNVRHPGGGPHAEET"
    "VLGNAAAAAAGEITTIVAVGDNNKGIIRPCGKCRGILLDLHPNCKAIVKDENGQPVAVPI"
    "SSLLPYPYKELTPEEQALIAAATATVNSIPASSRHNVAAAALSAKGQIFTGVNVYHPGGG"
    "PHAEETVLGNAAAANAGELTAIVCVGQGGAGIIAPCGRCRGILRELYPNVRCIVLDANG"
    "QPVAVPVASLLPPMTPAEEKLIELATATVNSIPPSDVHNVAAAALSADGRYFTGVNVRH"
    "PGGGPHAEQVVLGVAAAAAAGEITAIVAVARNGGGIIRPCGRCRGILKALHPNVRAIV"
    "KDENGKPVAKPIASLLP"
)

with torch.no_grad():
    output = model.infer(sequence)

int_keys = {"aatype", "residx_atom37_to_atom14", "residx_atom14_to_atom37", "residue_index"}
np_output = {}
for k, v in output.items():
    arr = v.to("cpu")
    if k in int_keys:
        np_output[k] = arr.long().numpy()
    else:
        np_output[k] = arr.float().numpy()

final_atom_positions = atom14_to_atom37(np_output["positions"][-1], np_output)
final_atom_mask = np_output["atom37_atom_exists"]

pred = Protein(
    aatype=np_output["aatype"][0],
    atom_positions=final_atom_positions[0],
    atom_mask=final_atom_mask[0],
    residue_index=np_output["residue_index"][0] + 1,
    b_factors=np_output["plddt"][0],
)

pdb_string = to_pdb(pred)

with open("demo/demo_b0_d0_pred.pdb", "w") as f:
    f.write(pdb_string)

print("pLDDT:", float(np_output["plddt"][0].mean()))
print("pTM:", float(np_output["ptm"]))
print("PDB saved to demo/demo_b0_d0_pred.pdb")
