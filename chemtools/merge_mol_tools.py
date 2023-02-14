from itertools import product
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem.rdchem import BondType
from .chemutils import get_smiles, copy_atom, sanitize
from copy import copy 

class Record:
    def __init__(self, merged_mol, root_mol, next_mol, anchor_nodes, next_mol_nodes, dgl_graphs=None):
        self.merged_mol = merged_mol
        self.root_mol = root_mol 
        self.next_mol = next_mol
        self.anchor_nodes = anchor_nodes
        self.next_mol_nodes = next_mol_nodes
        self.dgl_graphs = dgl_graphs

def unique(records):
    records_new = []
    smileses = set()
    for r in records:
        if get_smiles(r.merged_mol) not in smileses:
            smileses.add(get_smiles(r.merged_mol))
            records_new.append(r)
    return records_new


def attach_mol(ctr_mol, next_mol, cand, global_amap):
    amap = global_amap[-1]
    for atom in next_mol.GetAtoms():
        new_atom = copy_atom(atom)
        amap[atom.GetIdx()] = ctr_mol.AddAtom(new_atom)

    for bond in next_mol.GetBonds():
        a1 = amap[bond.GetBeginAtom().GetIdx()]
        a2 = amap[bond.GetEndAtom().GetIdx()]
        if ctr_mol.GetBondBetweenAtoms(a1, a2) is None:
            ctr_mol.AddBond(a1, a2, bond.GetBondType())
    
    ctr_mol.AddBond(cand[0], amap[cand[1]], BondType.SINGLE)

    return ctr_mol

def enum_assemble(scaffold_node, side_chain_node):
    return list(product(scaffold_node['attachment_pos'], side_chain_node['attachment_pos']))


def assemble(scaffold_node, side_chain_node):
    cands = enum_assemble(scaffold_node, side_chain_node)
    global_amap = [{atom.GetIdx(): atom.GetIdx() for atom in scaffold_node['mol'].GetAtoms()}, {}]

    backup_mol = Chem.RWMol(scaffold_node['mol'])
    records = []
    for scaffold_atom, side_chain_atom in cands:
        cur_mol = Chem.RWMol(backup_mol)

        new_global_amap = copy.deepcopy(global_amap)
        
        cur_mol = attach_mol(cur_mol, side_chain_node['mol'], [scaffold_atom, side_chain_atom], new_global_amap)
        new_mol = cur_mol.GetMol()
        new_mol = sanitize(new_mol)

        if new_mol is None: continue 
        
        records.append(Record(merged_mol=cur_mol,
                    root_mol=scaffold_node['mol'],
                    next_mol=side_chain_node['mol'],
                    anchor_nodes=[scaffold_atom, new_global_amap[-1][side_chain_atom]],
                    next_mol_nodes=list(new_global_amap[-1].values()), 
                    dgl_graphs=None))
        
    return records

