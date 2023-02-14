import pickle as pkl
import os
from rdkit import Chem
from chemtools.merge_mol_tools import Record, unique
from chemtools.merge_mol_tools import assemble
from chemtools.chemutils import get_smiles, get_mol, sanitize
from tqdm import tqdm


def generate_records(scaffold_node, side_chain_nodes):
    records = []
    for side_chain_node in tqdm(side_chain_nodes):
        records.extend(assemble(scaffold_node, side_chain_node))
    records = unique(records)
    return records

def get_anchor_node(merged_mol, root_mol):
    idx_matched_root_mol = set(merged_mol.GetSubstructMatch(root_mol))
    idx_not_matched_root_mol = set(range(merged_mol.GetNumAtoms())) - idx_matched_root_mol
    anchor_nodes = None
    for bond in merged_mol.GetBonds():
        if (bond.GetBeginAtomIdx() in idx_matched_root_mol and bond.GetEndAtomIdx() in idx_not_matched_root_mol) or (bond.GetEndAtomIdx() in idx_matched_root_mol and bond.GetBeginAtomIdx() in idx_not_matched_root_mol):
            anchor_nodes = [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
    assert anchor_nodes is not None 
    return anchor_nodes


if __name__ == '__main__':
    motif_nodes = pkl.load(open(
        'dataset_attack/motif/side_chain_nodes_1.pkl',
        'rb'))
    
    scaffold_nodes = pkl.load(open(
            'dataset_attack/scaffold/scaffolds_nodes_top8.pkl',
            'rb'))
    scaffold_node = scaffold_nodes[6]

    records = generate_records(scaffold_node, motif_nodes)

    for r in tqdm(records):
        r.merged_mol = Chem.MolFromSmiles(get_smiles(r.merged_mol))
        
    for r in tqdm(records):
        r.merged_mol = Chem.MolFromSmiles(get_smiles(r.merged_mol))
        mol_bk = Chem.MolFromSmiles(get_smiles(r.merged_mol))
        root_mol = Chem.MolFromSmiles(get_smiles(r.root_mol))
        anchor_nodes = get_anchor_node(mol_bk, root_mol)
        r.anchor_nodes = anchor_nodes

    # pkl.dump(records, open(f'dataset_attack/records/{scaffold_smiles}/{scaffold_smiles}_records.pkl', 'wb'))