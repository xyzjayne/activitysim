# ActivitySim
# See full license in LICENSE.txt.

import os
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
logger.addHandler(ch)

survey_dir = os.path.dirname('data/survey_data/')
output_dir = os.path.dirname('data/estimation_choices/')
configs_dir = os.path.dirname('../example/configs/')

surveys = {
    'households': 'survey_households.csv',
    'persons': 'survey_persons.csv',
    'tours': 'survey_tours.csv',
    'joint_tour_participants': 'survey_joint_tour_participants.csv'
}

outputs = {
    'households': 'override_households.csv',
    'persons': 'override_persons.csv',
    'tours': 'override_tours.csv',
    'joint_tour_participants': 'override_joint_tour_participants.csv'
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

    assert tours.start.isin(tdd_alts.start).all(), "not all tour starts in tdd_alts"
    assert tours.end.isin(tdd_alts.end).all(), "not all tour starts in tdd_alts"

    tdds = pd.merge(tours[['start', 'end']], tdd_alts, left_on=['start', 'end'], right_on=['start', 'end'], how='left')

    if tdds.tdd.isna().any():
        bad_tdds = tours[tdds.tdd.isna()]
        print("Bad tour start/end times:")
        print(bad_tdds)
        bug

    # print("tdd_alts\n",tdd_alts, "\n")
    # print("tours\n",tours[['start', 'end']])
    # print("tdds\n",tdds)
    return tdds.tdd


def infer_joint_tour_frequency(households, persons, tours):

    def read_alts():
        # right now this file just contains the start and end hour
        alts = \
            pd.read_csv(os.path.join(configs_dir, 'joint_tour_frequency_alternatives.csv'),
                        comment='#', index_col='alt')
        alts = alts.astype(np.int8)  # - NARROW
        return alts

    alts = read_alts()
    tour_types = list(alts.columns.values)

    assert(len(alts.index[(alts == 0).all(axis=1)]) == 1)  # should be one zero_tours alt
    zero_tours_alt = alts.index[(alts == 0).all(axis=1)].values[0]

    alts['joint_tour_frequency'] = alts.index
    joint_tours = tours[tours.tour_category == 'joint']

    num_tours = pd.DataFrame(index=households.index)
    for tour_type in tour_types:
        joint_tour_is_tour_type = (joint_tours.tour_type == tour_type)
        if joint_tour_is_tour_type.any():
            num_tours[tour_type] = joint_tours[joint_tour_is_tour_type].groupby('household_id').size().reindex(households.index).fillna(0)
        else:
            logger.warning("WARNING infer_joint_tour_frequency - no tours of type '%s'" % tour_type)
            num_tours[tour_type] = 0
    num_tours = num_tours.fillna(0).astype(int)

    # need to do index waltz because pd.merge doesn't preserve index in this case
    jtf = pd.merge(num_tours.reset_index(), alts, left_on=tour_types, right_on=tour_types, how='left').set_index(households.index.name)

    if jtf.joint_tour_frequency.isna().any():
        bad_tour_frequencies = jtf.joint_tour_frequency.isna()
        logger.warning("WARNING Bad joint tour frequencies\n\n")
        logger.warning("\nWARNING Bad joint tour frequencies: num_tours\n%s\n", num_tours[bad_tour_frequencies])
        logger.warning("\nWARNING Bad joint tour frequencies: num_tours\n%s\n", joint_tours[joint_tours.household_id.isin(bad_tour_frequencies.index)])
        bug

    logger.info("infer_joint_tour_frequency: %s households with joint tours", (jtf.joint_tour_frequency != zero_tours_alt).sum())

    return jtf.joint_tour_frequency


def infer_joint_tour_composition(households, persons, tours, joint_tour_participants):
    """
    assign joint_tours a 'composition' column ('adults', 'children', or 'mixed')
    depending on the composition of the joint_tour_participants
    """
    joint_tours = tours[tours.tour_category == 'joint'].copy()

    joint_tour_participants = \
        pd.merge(joint_tour_participants, persons,
                 left_on='person_id', right_index=True, how='left')


    # FIXME - computed by asim annotate persons - not needed if embeded in asim and called just-in-time
    if 'adult' not in joint_tour_participants:
        joint_tour_participants['adult'] = (joint_tour_participants.age >= 18)

    tour_has_adults = \
        joint_tour_participants[joint_tour_participants.adult]\
        .groupby('tour_id').size()\
        .reindex(joint_tours.tour_id).fillna(0) > 0

    tour_has_children = \
        joint_tour_participants[~joint_tour_participants.adult]\
        .groupby('tour_id').size()\
        .reindex(joint_tours.tour_id).fillna(0) > 0

    assert (tour_has_adults | tour_has_children).all()

    joint_tours['composition'] = np.where(tour_has_adults, np.where(tour_has_children, 'mixed', 'adults'), 'children')

    return joint_tours.composition.reindex(tours.index).fillna('').astype(str)



households = pd.read_csv(os.path.join(survey_dir, surveys['households']), index_col='household_id')
persons = pd.read_csv(os.path.join(survey_dir, surveys['persons']), index_col='person_id')
tours = pd.read_csv(os.path.join(survey_dir, surveys['tours']))
joint_tour_participants = pd.read_csv(os.path.join(survey_dir, surveys['joint_tour_participants']))

persons['cdap_activity'] = infer_cdap_activity(households, persons, tours)

persons['mandatory_tour_frequency'] = infer_mandatory_tour_frequency(households, persons, tours)

households['joint_tour_frequency'] = infer_joint_tour_frequency(households, persons, tours)

tours['composition'] = infer_joint_tour_composition(households, persons, tours, joint_tour_participants)

tours['tdd'] = infer_tour_scheduling(households, persons, tours)

households.to_csv(os.path.join(output_dir, outputs['households']), index=True)
persons.to_csv(os.path.join(output_dir, outputs['persons']), index=True)
tours.to_csv(os.path.join(output_dir, outputs['tours']), index=False)