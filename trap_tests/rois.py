import pandas as pd
import numpy as np
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

limit = 500
healthy_subjects[:limit]
depressive_subjects[:limit]
depressive_covariate[:limit]

rois = {'hippocampus': ['Hippocampus_L', 'Hippocampus_R'],
        'amygdala': ['Amygdala_L', 'Amydala_R'],
        'putamen': ['Putamen_L', 'Putamen_R'],
        'frontal': ['Frontal_Sup_Orb_L', 'Frontal_Sub_Orb_R'],
        'acc': ['Cingulum_Ant_L', 'Cingulum_Ant_R']}

data_folder = './data'
for roi_name, roi in rois.items():
    healthy_filename = os.path.join(data_folder, "{}_healthy.npy".format(roi_name))
    depressed_filename = os.path.join(data_folder, "{}_depressed.npy".format(roi_name))
    if os.path.exists(healthy_filename) and os.path.exists(depressed_filename):
        healthy_data = np.load(healthy_filename)
        depressive_data = np.load(depressed_filename)

    else:
        neuro_branch = NeuroBranch('NeuroBranch', nr_of_processes=1)

        # resample images to a desired voxel size - this also works with voxel_size as hyperparameter
        # it's also very reasonable to define a batch size for a large number of subjects
        neuro_branch += PipelineElement('ResampleImages', voxel_size=5, batch_size=20)

        # now, apply a brain atlas and extract 4 ROIs
        # set "extract_mode" to "vec" so that all voxels within these ROIs are vectorized and concatenated
        neuro_branch += PipelineElement('BrainAtlas', hyperparameters={},
                                        rois=['Hippocampus_L', 'Hippocampus_R', 'Amygdala_L', 'Amygdala_R', 'Putamen_L', 'Putamen_R'],
                                        atlas_name="AAL", extract_mode='vec', batch_size=20)
        healthy_data = neuro_branch.transform(healthy_subjects['NeuroDerivatives_r1184_gray_matter'].to_list())
        depressive_data = neuro_branch.transform(depressive_subjects['NeuroDerivatives_r1184_gray_matter'].to_list())

        np.save(healthy_filename, healthy_data)
        np.save(depressed_filename, depressive_data)

    cpca = CPCA(preprocess_with_pca_dim=10000)
    transformed_data = cpca.fit_transform(depressive_data, healthy_data, depressive_covariate)
    cpca.plot(depressive_data, depressive_covariate)

debug = True