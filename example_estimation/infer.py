# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import numpy as np
import pandas as pd

survey_dir = os.path.dirname('data/survey_data/')
output_dir = os.path.dirname('data/estimation_choices/')
configs_dir = os.path.dirname('../example/configs/')

surveys = {
    'households': 'survey_households.csv',
    'persons': 'survey_persons.csv',
    'tours': 'survey_tours.csv'
}

outputs = {
    'households': 'override_households.csv',
    'persons': 'override_persons.csv',
    'tours': 'override_tours.csv'
}


def infer_mandatory_tour_frequency(households, persons, tours):

    num_work_tours = \
        tours[tours.tour_type == 'work'].\
            groupby('person_id').size().reindex(persons.index).fillna(0).astype(np.int8)

    num_school_tours = \
        tours[tours.tour_type == 'school'].\
            groupby('person_id').size().reindex(persons.index).fillna(0).astype(np.int8)

    mtf = {
        0: '',
        1: 'work1',
        2: 'work2',
        10: 'school1',
        20: 'school2',
        11: 'work_and_school'
    }

    mandatory_tour_frequency = (num_work_tours + num_school_tours*10).map(mtf)
    return mandatory_tour_frequency


def infer_cdap_activity(households, persons, tours):

    mandatory_tour_types = ['work', 'school']
    non_mandatory_tour_types = ['escort', 'shopping', 'othmaint', 'othdiscr', 'eatout', 'social']

    cdap_activity = pd.Series('H', index=persons.index)

    num_mandatory_tours = \
        tours[tours.tour_type.isin(mandatory_tour_types)].\
            groupby('person_id').size().\
            reindex(cdap_activity.index).fillna(0).astype(np.int8)

    num_non_mandatory_tours = \
        tours[tours.tour_type.isin(non_mandatory_tour_types)].\
            groupby('person_id').size().\
            reindex(cdap_activity.index).fillna(0).astype(np.int8)

    cdap_activity = cdap_activity.where(num_mandatory_tours == 0, 'M')
    cdap_activity = cdap_activity.where((cdap_activity == 'M') | (num_non_mandatory_tours == 0), 'N')

    return cdap_activity

def infer_tour_scheduling(households, persons, tours):
    # given start and end periods, infer tdd

    def read_tdd_alts():
        # right now this file just contains the start and end hour
        tdd_alts = pd.read_csv(os.path.join(configs_dir, 'tour_departure_and_duration_alternatives.csv'))
        tdd_alts['duration'] = tdd_alts.end - tdd_alts.start
        tdd_alts = tdd_alts.astype(np.int8)  # - NARROW

        tdd_alts['tdd'] = tdd_alts.index
        return tdd_alts

    tdd_alts = read_tdd_alts()

    #assert tours.start.isin(tdd_alts.start).all(), "not all tour starts in tdd_alts"
    #assert tours.end.isin(tdd_alts.end).all(), "not all tour starts in tdd_alts"

    tdds = pd.merge(tours[['start', 'end']], tdd_alts, left_on=['start', 'end'], right_on=['start', 'end'], how='left')

    if tdds.tdd.isna().any():
        bad_tdds = tours[tdds.tdd.isna()]
        print("Bad tour start/end times:")
        print(bad_tdds)
        bug

    print("tdd_alts\n",tdd_alts, "\n")
    print("tours\n",tours[['start', 'end']])
    print("tdds\n",tdds)
    return tdds.tdd

households = pd.read_csv(os.path.join(survey_dir, surveys['households']), index_col='household_id')
persons = pd.read_csv(os.path.join(survey_dir, surveys['persons']), index_col='person_id')
tours = pd.read_csv(os.path.join(survey_dir, surveys['tours']))

persons['cdap_activity'] = infer_cdap_activity(households, persons, tours)

persons['mandatory_tour_frequency'] = infer_mandatory_tour_frequency(households, persons, tours)

tours['tdd'] = infer_tour_scheduling(households, persons, tours)

households.to_csv(os.path.join(output_dir, outputs['households']), index=True)
persons.to_csv(os.path.join(output_dir, outputs['persons']), index=True)
tours.to_csv(os.path.join(output_dir, outputs['tours']), index=False)