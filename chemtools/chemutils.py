import rdkit
import rdkit.Chem as Chem
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions

MST_MAX_WEIGHT = 100 
MAX_NCAND = 2000

def set_atommap(mol, num=0):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None
    Chem.Kekulize(mol)
    return mol

def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)

def decode_stereo(smiles2D):
    mol = Chem.MolFromSmiles(smiles2D)
    dec_isomers = list(EnumerateStereoisomers(mol))

    dec_isomers = [Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=True)) for mol in dec_isomers]
    smiles3D = [Chem.MolToSmiles(mol, isomericSmiles=True) for mol in dec_isomers]

    chiralN = [atom.GetIdx() for atom in dec_isomers[0].GetAtoms() if int(atom.GetChiralTag()) > 0 and atom.GetSymbol() == "N"]
    if len(chiralN) > 0:
        for mol in dec_isomers:
            for idx in chiralN:
                mol.GetAtomWithIdx(idx).SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
            smiles3D.append(Chem.MolToSmiles(mol, isomericSmiles=True))

    return smiles3D

def sanitize(mol):
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        return None
    return mol

def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol

def get_clique_mol(mol, atoms):
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol) #We assume this is not None
    return new_mol


def atom_equal(a1, a2):
    return a1.GetSymbol() == a2.GetSymbol() and a1.GetFormalCharge() == a2.GetFormalCharge()









if __name__ == "__main__":
    import sys
    # from mol_tree import MolTree
    # lg = rdkit.RDLogger.logger() 
    # lg.setLevel(rdkit.RDLogger.CRITICAL)
    
    # smiles = ["O=C1[C@@H]2C=C[C@@H](C=CC2)C1(c1ccccc1)c1ccccc1","O=C([O-])CC[C@@]12CCCC[C@]1(O)OC(=O)CC2", "ON=C1C[C@H]2CC3(C[C@@H](C1)c1ccccc12)OCCO3", "C[C@H]1CC(=O)[C@H]2[C@@]3(O)C(=O)c4cccc(O)c4[C@@H]4O[C@@]43[C@@H](O)C[C@]2(O)C1", 'Cc1cc(NC(=O)CSc2nnc3c4ccccc4n(C)c3n2)ccc1Br', 'CC(C)(C)c1ccc(C(=O)N[C@H]2CCN3CCCc4cccc2c43)cc1', "O=c1c2ccc3c(=O)n(-c4nccs4)c(=O)c4ccc(c(=O)n1-c1nccs1)c2c34", "O=C(N1CCc2c(F)ccc(F)c2C1)C1(O)Cc2ccccc2C1"]

    # def tree_test():
    #     for s in sys.stdin:
    #         s = s.split()[0]
    #         tree = MolTree(s)
    #         print '-------------------------------------------'
    #         print s
    #         for node in tree.nodes:
    #             print node.smiles, [x.smiles for x in node.neighbors]

    # def decode_test():
    #     wrong = 0
    #     for tot,s in enumerate(sys.stdin):
    #         s = s.split()[0]
    #         tree = MolTree(s)
    #         tree.recover()

    #         cur_mol = copy_edit_mol(tree.nodes[0].mol)
    #         global_amap = [{}] + [{} for node in tree.nodes]
    #         global_amap[1] = {atom.GetIdx():atom.GetIdx() for atom in cur_mol.GetAtoms()}

    #         dfs_assemble(cur_mol, global_amap, [], tree.nodes[0], None)

    #         cur_mol = cur_mol.GetMol()
    #         cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
    #         set_atommap(cur_mol)
    #         dec_smiles = Chem.MolToSmiles(cur_mol)

    #         gold_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(s))
    #         if gold_smiles != dec_smiles:
    #             print gold_smiles, dec_smiles
    #             wrong += 1
    #         print wrong, tot + 1

    # def enum_test():
    #     for s in sys.stdin:
    #         s = s.split()[0]
    #         tree = MolTree(s)
    #         tree.recover()
    #         tree.assemble()
    #         for node in tree.nodes:
    #             if node.label not in node.cands:
    #                 print tree.smiles
    #                 print node.smiles, [x.smiles for x in node.neighbors]
    #                 print node.label, len(node.cands)

    # def count():
    #     cnt,n = 0,0
    #     for s in sys.stdin:
    #         s = s.split()[0]
    #         tree = MolTree(s)
    #         tree.recover()
    #         tree.assemble()
    #         for node in tree.nodes:
    #             cnt += len(node.cands)
    #         n += len(tree.nodes)
    #         #print cnt * 1.0 / n
    
    # count()
