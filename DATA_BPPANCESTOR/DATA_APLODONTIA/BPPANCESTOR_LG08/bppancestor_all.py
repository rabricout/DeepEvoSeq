from pathlib import Path
from ete3 import Tree
from Bio import SeqIO
import subprocess
import shutil

ali_folder = Path("./DATA_ALIGNED/")
tree_folder = Path("./DATA_TREEFILES/")
out_folder = Path("./DATA_BPP_ALL")
a1_out_folder = Path("./DATA_A1")
tmp_folder = Path("./tmp")

out_folder.mkdir(exist_ok=True)
a1_out_folder.mkdir(exist_ok=True)
tmp_folder.mkdir(exist_ok=True)

for fasta_file in ali_folder.glob("*.fasta"):
    print('> Doing', fasta_file)
    base = fasta_file.stem
    tree_file = tree_folder / f"{base}.fasta.treefile"
    

    if tree_file.exists():
        print(tree_file)
        conf_content = f"""
input.sequence.file = {fasta_file}
input.sequence.format = Fasta(strictNames=yes, extended=yes)
alphabet=Protein
input.sequence.sites_to_use=all
input.tree.file = {tree_file}
input.tree.format = Newick
model = LG08
output.sequence.file = {out_folder / base}_ancestor.fasta
output.sequence.format = Fasta
rooting = Outgroup(first_sequence)
"""

        conf_content_tree = f"""
input.sequence.file = {fasta_file}
input.sequence.format = Fasta(strictNames=yes, extended=yes)
alphabet=Protein
input.sequence.sites_to_use=all
input.tree.file = {tree_file}
input.tree.format = Newick
model = LG08
output.tree_ids.file = {out_folder / base}_tree_with_nodes.nwk
rooting = Outgroup(first_sequence)
"""

        # For some reason, bppancestor cannot create both output sequence file and tree_ids file, so we run both separately

        conf_path = tmp_folder / f"{base}.conf"
        conf_path.write_text(conf_content)
        subprocess.run(["bppancestor", f"param={conf_path}"], check=True)

        conf_path_tree = tmp_folder / f"{base}_tree.conf"
        conf_path_tree.write_text(conf_content_tree)
        subprocess.run(["bppancestor", f"param={conf_path_tree}"], check=True)

        tree = Tree(f"{out_folder / base}_tree_with_nodes.nwk", format=1)
        leaves = [leaf for leaf in tree.get_leaves() if "Aplodontia" in leaf.name]
        if len(leaves)==1:
            closest_ancestor_id = leaves[0].up.name
        else:
            print("Multiples leaves")
            input('ERROR, press to continue')
        records = list(SeqIO.parse(fasta_file, "fasta"))
        for record in SeqIO.parse(f"{out_folder / base}_ancestor.fasta", "fasta"):
            if record.id == closest_ancestor_id:
                record.id = 'A1_closest_aplodontia_ancestor'
                records.append(record)
        with open(f"{a1_out_folder / base}.fasta", "w") as out:
            SeqIO.write(records, out, "fasta")
shutil.rmtree(tmp_folder)