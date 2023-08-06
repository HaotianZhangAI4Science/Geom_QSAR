from rdkit import Chem
import pickle


def read_sdf(sdf_file):
    supp = Chem.SDMolSupplier(sdf_file)
    mols_list = [i for i in supp]
    return mols_list

def write_sdf(mol_list,file):
    writer = Chem.SDWriter(file)
    for i in mol_list:
        writer.write(i)
    writer.close()

def read_pkl(file):
    with open(file,'rb') as f:
        data = pickle.load(f)
    return data

def write_pkl(list,file):
    with open(file,'wb') as f:
        pickle.dump(list,f)
        print('pkl file saved at {}'.format(file))