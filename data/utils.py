import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from rdkit.Chem import MolFromSmiles


#===============================================================
def onehot_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


#===============================================================
def onehot_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

#===============================================================
# Get node's feature 
def atom_attr(mol, explicit_H=True, use_chirality=True):
    '''Each atom will be encoded using one-hot encoder based on its physiochemical properties
    input: mol 
    output: Nx39 (N: Number of heavy atom in compound, total length of vector one-hot=39 (Each vector indicate different properties) )
    '''
    feat = []
    for i, atom in enumerate(mol.GetAtoms()):
        results = onehot_encoding_unk(
            atom.GetSymbol(),
            ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At', 'other'
             ]) + onehot_encoding(atom.GetDegree(),
                                  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  onehot_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                      Chem.rdchem.HybridizationType.SP3D2, 'other'
                  ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if not explicit_H:
            results = results + onehot_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4])
        if use_chirality:
            try:
                results = results + onehot_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                results = results + [0, 0] + [atom.HasProp('_ChiralityPossible')]
        feat.append(results)
    return np.array(feat)

#===============================================================
# Get edge's index 
def bond_attr(mol, use_chirality=True):
    feat = []
    index = []
    n = mol.GetNumAtoms()
    for i in range(n):
        for j in range(n):
            if i != j:
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is not None:
                    index.append([i, j])
    return np.array(index), np.array(feat)


#===============================================================
# Encode mol to graph
def mol2graph(mol):
    if mol is None: return None
    node_attr = atom_attr(mol)
    edge_index, edge_attr = bond_attr(mol)
    # pos = torch.FloatTensor(geom)
    data = Data(
        x=torch.FloatTensor(node_attr),
        # pos=pos,
        edge_index=torch.LongTensor(edge_index).t(),
        edge_attr=torch.FloatTensor(edge_attr),
        y=None  # None as a placeholder
    )
    return data


#===============================================================
# Encode
def get_encode(smilesList, target):
    data_list = []
    for i, smi in enumerate(tqdm(smilesList)):
        mol = MolFromSmiles(smi)
        data = mol2graph(mol)
        label = []
        if data is not None:
            for task in range(len(target)):
                label.append(target[task][i])
            data.y = torch.LongTensor([label])
            data_list.append(data)
    return data_list


#===============================================================
# MultiDataset
class MultiDataset(InMemoryDataset):
    def __init__(self, root, data_list, dataset, transform=None, pre_transform=None, pre_filter=None):
        self.dataset = dataset
        self.data_list = data_list
        self.weights = 0
        super(MultiDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # os.remove(self.processed_paths[0])
    #---------------------------------------
    @property
    def raw_file_names(self):
        return ['{}.csv'.format(self.dataset)]
    #---------------------------------------
    @property
    def processed_file_names(self):
        return ['{}.pt'.format(self.dataset)]
    #---------------------------------------
    def download(self):
        pass
    #---------------------------------------
    def process(self):
        data, slices = self.collate(self.data_list)
        torch.save((data, slices), self.processed_paths[0])

