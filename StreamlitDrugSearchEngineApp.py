import mols2grid as m2g
import pandas as pd
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import Descriptors

# App title and concise subtitle
img = Image.open('dopamine_chemical_structure.png')

col1, col2, col3 = st.columns([3, 1, 2])  # st.columns([3, 1, 2]) creates 3 columns
                                          # where the first column is 3 times the width of the second 
                                          # and the last column is 2 times that width                        

with col1:
    st.header('Drug search engine')
with col2:
    st.write('')
with col3:
    st.image(img, width=200)

st.info('*This Streamlit app allows you to search for the most "druglike" molecules basing on the "Rule of five"*')


# Open and preprocess data files
@st.cache(allow_output_mutation = True)
def prep_file(fin_name, fout_name):
    with open(fin_name) as fin, open(fout_name, 'w') as fout:
        for line in fin:
            fout.write(line.replace('\t', ','))
    

@st.cache(allow_output_mutation = True)
def read_dataset(fout_name):
    df = pd.read_csv(fout_name, on_bad_lines = 'skip').dropna()

    return df

# Calculate molecules' descriptors
def calculate_mol_weight(smiles):
    molecule = Chem.MolFromSmiles(smiles)

    return Descriptors.ExactMolWt(molecule)

def calculate_log_p(smiles):
    molecule =  Chem.MolFromSmiles(smiles)

    return Descriptors.MolLogP(molecule)

def calculate_num_h_donors(smiles):
    molecule = Chem.MolFromSmiles(smiles)

    return Descriptors.NumHDonors(molecule)

def calculate_num_h_acceptors(smiles):
    molecule = Chem.MolFromSmiles(smiles)

    return Descriptors.NumHAcceptors(molecule)


# Read dataset
fin_name, fout_name = 'drugs.txt', 'drugs.csv'

prep_file(fin_name, fout_name)
df = read_dataset(fout_name).copy()

# Add calculated columns containing descriptors to dataframe
df['Molecular weight'] = df.apply(lambda x: calculate_mol_weight(x['smiles']), axis = 'columns')
df['LogP'] = df.apply(lambda x: calculate_log_p(x['smiles']), axis = 'columns')
df['Hydrogen bond donors'] = df.apply(lambda x: calculate_num_h_donors(x['smiles']), axis = 'columns')
df['Hydrogen bond acceptors'] = df.apply(lambda x: calculate_num_h_acceptors(x['smiles']), axis = 'columns')

# Drop unnecessary colums and relabel remaining ones
df = df.drop('cns_drug', axis = 'columns').rename(columns={'generic_name': 'Drug INN (international nonproprietary name)', 'smiles': 'SMILES'})

# Streamlit sidebar title and subtitle
st.sidebar.header('Set cutoff values')
st.sidebar.write('*Set threshold value for each descriptor*')

# Sidebar sliders
mol_weight_cutoff = st.sidebar.slider('Molecular weight', 0, 1000, 500, 10)  # Paremeters respectively: label, min_value, max_value, default_value, step
log_p_cutoff = st.sidebar.slider('LogP', -10, 10, 5, 1)
num_h_donors_cutoff = st.sidebar.slider('Hydrogen bond donors', 0, 15, 5, 1)
num_h_acceptors_cutoff = st.sidebar.slider('Hydrogen bond acceptors', 0, 20, 10, 1)

# Calculate resultant dataframe for all sliders' settings
df_output1 = df[df['Molecular weight'] < mol_weight_cutoff]
df_output2 = df_output1[df_output1['LogP'] < log_p_cutoff]
df_output3 = df_output2[df_output2['Hydrogen bond donors'] < num_h_donors_cutoff]
df_output4 = df_output3[df_output3['Hydrogen bond acceptors'] < num_h_acceptors_cutoff]

df_shape = st.sidebar.write('Dataframe (rows, columns): ', df_output4.shape)

if st.checkbox('Show/hide dataframe'):
    st.write(df_output4)

# Display the results
raw_html = m2g.display(df_output4, fixedBondLength = 20, subset = ['Drug INN (international nonproprietary name)', 
                                                                   'Molecular weight', 'LogP', 'img'],

                                                         style = {'Molecular weight': lambda x: 'color: lightgreen' if x < 500 else 'color: black', 
                                                                  'LogP': lambda x: 'color: lightgreen' if x < 5 else 'color: black'},
                  
                                                         transform = {'Molecular weight': lambda x: f'{x:.2f}',
                                                                      'LogP': lambda x: f'{x:.2f}'})._repr_html_()

components.html(raw_html, width = 900, height = 1100, scrolling = False)

