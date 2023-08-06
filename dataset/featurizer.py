import numpy as np
import rdkit
from rdkit import Chem
from copy import deepcopy
import torch 

atomTypes = ['H', 'C', 'B', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Br', 'I'] #12
formalCharge = [-1, -2, 1, 2, 0] 
hybridization = [
    rdkit.Chem.rdchem.HybridizationType.S,
    rdkit.Chem.rdchem.HybridizationType.SP,
    rdkit.Chem.rdchem.HybridizationType.SP2,
    rdkit.Chem.rdchem.HybridizationType.SP3,
    rdkit.Chem.rdchem.HybridizationType.SP3D,
    rdkit.Chem.rdchem.HybridizationType.SP3D2,
    rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED,
]
num_single_bonds = [0,1,2,3,4,5,6]
num_double_bonds = [0,1,2,3,4]
num_triple_bonds = [0,1,2]
num_aromatic_bonds = [0,1,2,3,4]
bondTypes = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']

def adjacency_to_undirected_edge_index(adj):
    adj = np.triu(np.array(adj, dtype = int)) #keeping just upper triangular entries from sym matrix
    array_adj = np.array(np.nonzero(adj), dtype = int) #indices of non-zero values in adj matrix
    edge_index = np.zeros((2, 2*array_adj.shape[1]), dtype = int) #placeholder for undirected edge list
    edge_index[:, ::2] = array_adj
    edge_index[:, 1::2] = np.flipud(array_adj)
    return edge_index

def one_hot_embedding(value, options):
    embedding = [0]*(len(options) + 1)
    index = options.index(value) if value in options else -1
    embedding[index] = 1
    return embedding

def getNodeFeatures(list_rdkit_atoms):
    '''
    Input: list of rdkit atoms
    Output: node features 
    node_feat_dim = 12 + 1 + 5 + 1 + 1 + 1 + 8 + 6 + 4 + 6 = 45
    atom_types feature (13), formal charge (6), aromatic (1), atomic mass (1), Bond related features (24)
    '''
    F_v = (len(atomTypes)+1) # 12 +1 
    F_v += (len(formalCharge)+1) # 5 + 1
    F_v += (1 + 1) 
    
    F_v += 8
    F_v += 6
    F_v += 4
    F_v += 6
    
    node_features = np.zeros((len(list_rdkit_atoms), F_v))
    for node_index, node in enumerate(list_rdkit_atoms):
        features = one_hot_embedding(node.GetSymbol(), atomTypes) # atom symbol, dim=12 + 1 
        features += one_hot_embedding(node.GetFormalCharge(), formalCharge) # formal charge, dim=5+1 
        features += [int(node.GetIsAromatic())] # whether atom is part of aromatic system, dim = 1
        features += [node.GetMass()  * 0.01] # atomic mass / 100, dim=1
        
        atom_bonds = np.array([b.GetBondTypeAsDouble() for b in node.GetBonds()])
        N_single = int(sum(atom_bonds == 1.0) + node.GetNumImplicitHs() + node.GetNumExplicitHs())
        N_double = int(sum(atom_bonds == 2.0))
        N_triple = int(sum(atom_bonds == 3.0))
        N_aromatic = int(sum(atom_bonds == 1.5))
        
        features += one_hot_embedding(N_single, num_single_bonds)
        features += one_hot_embedding(N_double, num_double_bonds)
        features += one_hot_embedding(N_triple, num_triple_bonds)
        features += one_hot_embedding(N_aromatic, num_aromatic_bonds)
        
        node_features[node_index,:] = features
        
    return np.array(node_features, dtype = np.float32)

def getEdgeFeatures(list_rdkit_bonds):
    '''
    Input: list of rdkit bonds
    Return: undirected edge features

    bondTypes = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'] + 1
    edge_feautre_dim = 5
    '''
    F_e = (len(bondTypes)+1) #+ 1 + (4+1)
    
    edge_features = np.zeros((len(list_rdkit_bonds)*2, F_e))
    for edge_index, edge in enumerate(list_rdkit_bonds):
        features = one_hot_embedding(str(edge.GetBondType()), bondTypes) # dim=4+1

        # Encode both directed edges to get undirected edge
        edge_features[2*edge_index: 2*edge_index+2, :] = features
        
    return np.array(edge_features, dtype = np.float32)

def featurize_mol(mol):
    '''
    featurize mol 
    '''
    if type(mol) == Chem.rdchem.Mol:
        mol_ = deepcopy(mol)
    elif type(mol) == str:
        if mol[-3:] == 'mol' or mol[-3:] == 'sdf':
            mol_ = Chem.MolFromMolFile(mol)
        elif mol[-4:] == 'mol2':
            mol_ = Chem.MolFromMol2File(mol)
        else:
            mol_ = Chem.MolFromSmiles(mol)

    Chem.SanitizeMol(mol_)
    adj = Chem.GetAdjacencyMatrix(mol_)
    edge_index = adjacency_to_undirected_edge_index(adj)
    bonds = []
    for b in range(int(edge_index.shape[1]/2)):
        bond_index = edge_index[:,::2][:,b]
        bond = mol_.GetBondBetweenAtoms(int(bond_index[0]), int(bond_index[1]))
        bonds.append(bond)
    edge_features = torch.tensor(getEdgeFeatures(bonds)) # PRECOMPUTE
    edge_index = torch.tensor(edge_index)

    atoms = Chem.rdchem.Mol.GetAtoms(mol_)
    node_features = torch.tensor(getNodeFeatures(atoms))
    xyz = torch.tensor(mol_.GetConformer().GetPositions(), dtype=torch.float32)
    element = torch.tensor(np.array([i.GetAtomicNum() for i in mol_.GetAtoms()]))

    data = {
        'element': element,
        'pos': xyz,
        'bond_index': edge_index,
        'bond_feature': edge_features,
        'atom_feature': node_features
    }
    return data

def featurize_frag(mol):
    mol_ = deepcopy(mol)
    Chem.SanitizeMol(mol_)
    adj = Chem.GetAdjacencyMatrix(mol_)
    edge_index = adjacency_to_undirected_edge_index(adj)
    bonds = []
    for b in range(int(edge_index.shape[1]/2)):
        bond_index = edge_index[:,::2][:,b]
        bond = mol_.GetBondBetweenAtoms(int(bond_index[0]), int(bond_index[1]))
        bonds.append(bond)
    edge_features = torch.tensor(getEdgeFeatures(bonds)) # PRECOMPUTE
    edge_index = torch.tensor(edge_index)
    atoms = Chem.rdchem.Mol.GetAtoms(mol_)
    node_features = torch.tensor(getNodeFeatures(atoms))
    return node_features, edge_index, edge_features
    
if __name__ == '__main__':
    mol = mols[0]
    mol_dict = featurize_mol(mol)