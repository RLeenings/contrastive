import sys
import pandas as pd
import os
from photonai.base import PipelineElement
from photonai_neuro.brain_atlas import AtlasLibrary
from contrastive import CPCA

root_folder = '/spm-data/Scratch/spielwiese_nils_winter/TrapDB/FOR2107/'
AtlasLibrary().list_rois('AAL')  # 'Schaefer2018_100Parcels_7Networks'

data = pd.read_csv(os.path.join(root_folder, 'FOR2107_DF1-3_220120_neuroimaging.csv'),
                   usecols=['DO_clinical', 'DO_T1', 'DO_FM_dropout_3mm',
                            'gender', 'age', 'hospitalization',
                            'research_diagnosis', 'NeuroDerivatives_r1184_gray_matter'])
data = data[data['DO_clinical'] == False]
data = data[data['DO_T1'] == False]
data = data[data['NeuroDerivatives_r1184_gray_matter'].notna()]
data['file_exists'] = [os.path.exists(mwp) for mwp in data['NeuroDerivatives_r1184_gray_matter']]
data = data[data['file_exists'] == True]

healthy_subjects = data[data['research_diagnosis'] == 'HC']
depressive_idx = data['research_diagnosis'] == 'MDD'
depressive_subjects = data[depressive_idx]
depressive_covariate = data[depressive_idx]['gender'].to_list()

limit = 10
healthy_subjects[:limit]
depressive_subjects[:limit]
depressive_covariate[:limit]

rois = [['Hippocampus_L', 'Hippocampus_R'],
        ['Amygdala_L', 'Amydala_R'],
        ['Putamen_L', 'Putamen_R'],
        ['Frontal_Sup_Orb_L', 'Frontal_Sub_Orb_R'],
        ['Cingulum_Ant_L', 'Cingulum_Ant_R']]

for roi in rois:
    atlas = PipelineElement('BrainAtlas',
                            rois=roi,
                            atlas_name="AAL", extract_mode='vec', batch_size=20)
    healthy_data = atlas.transform(healthy_subjects['NeuroDerivatives_r1184_gray_matter'].to_list())
    depressive_data = atlas.transform(depressive_subjects['NeuroDerivatives_r1184_gray_matter'].to_list())
    cpca = CPCA()
    transformed_data = cpca.fit_transform(depressive_data, healthy_data, depressive_covariate)
    cpca.plot(transformed_data, depressive_covariate)

debug = True