# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import os
import logging

import pandas as pd
import pandas.util.testing as pdt
import pytest
import yaml

from activitysim.core import random
from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import inject
from activitysim.core import config

# set the max households for all tests (this is to limit memory use on travis)
HOUSEHOLDS_SAMPLE_SIZE = 100

# household with mandatory, non mandatory, atwork_subtours, and joint tours
HH_ID = 1931818

# [ 728370 1234067 1402924 1594625 1595333 1747572 1896849 1931818 2222690 2344951 2677154]

# SKIP_FULL_RUN = True
SKIP_FULL_RUN = False


def setup_dirs(configs_dir):

    inject.add_injectable("configs_dir", configs_dir)

    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    inject.add_injectable("output_dir", output_dir)

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    inject.add_injectable("data_dir", data_dir)

    inject.clear_cache()

    tracing.config_logger()

    tracing.delete_output_files('csv')
    tracing.delete_output_files('txt')
    tracing.delete_output_files('yaml')


def teardown_function(func):
    inject.clear_cache()
    inject.reinject_decorated_tables()


def close_handlers():

    loggers = logging.Logger.manager.loggerDict
    for name in loggers:
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True
        logger.setLevel(logging.NOTSET)


def inject_settings(configs_dir, **kwargs):

    with open(os.path.join(configs_dir, 'settings.yaml')) as f:
        settings = yaml.load(f)

        for k in kwargs:
            settings[k] = kwargs[k]

        inject.add_injectable("settings", settings)

    return settings


def test_rng_access():

    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')

    setup_dirs(configs_dir)

    inject.add_injectable('rng_base_seed', 0)

    pipeline.open_pipeline()

    rng = pipeline.get_rn_generator()

    assert isinstance(rng, random.Random)

    pipeline.close_pipeline()
    inject.clear_cache()


def regress_mini_auto():

    # regression test: these are among the first 10 households in households table
    hh_ids = [257127, 566664, 2344918, 823865]
    choices = [0, 1, 2, 0]
    expected_choice = pd.Series(choices, index=pd.Index(hh_ids, name="household_id"),
                                name='auto_ownership')

    auto_choice = pipeline.get_table("households").auto_ownership
    print("auto_choice\n", auto_choice.head(4))

    auto_choice = auto_choice.reindex(hh_ids)

    """
    auto_choice
     household_id
    257127     0
    566664     1
    2344918    2
    823865     0
    Name: auto_ownership, dtype: int64
    """
    pdt.assert_series_equal(auto_choice, expected_choice)


def regress_mini_mtf():

    mtf_choice = pipeline.get_table("persons").mandatory_tour_frequency

    # these choices are for pure regression - their appropriateness has not been checked
    per_ids = [29024, 107599, 108875]
    choices = ['school1', 'work1', 'work1']
    expected_choice = pd.Series(choices, index=pd.Index(per_ids, name='person_id'),
                                name='mandatory_tour_frequency')

    print("mtf_choice\n", mtf_choice[mtf_choice != ''].head(5))
    """
    mtf_choice
     person_id           
    29024     school1        
    107599      work1
    108875      work1
    109095      work1
    109192      work1
    Name: mandatory_tour_frequency, dtype: object
    """
    pdt.assert_series_equal(mtf_choice.reindex(per_ids), expected_choice)


def test_mini_pipeline_run():

    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')

    setup_dirs(configs_dir)

    inject_settings(configs_dir,
                    households_sample_size=HOUSEHOLDS_SAMPLE_SIZE,
                    # use_shadow_pricing=True
                    )

    _MODELS = [
        'initialize_landuse',
        'compute_accessibility',
        'initialize_households',
        'school_location',
        'workplace_location',
        'auto_ownership_simulate'
    ]

    pipeline.run(models=_MODELS, resume_after=None)

    regress_mini_auto()

    pipeline.run_model('cdap_simulate')
    pipeline.run_model('mandatory_tour_frequency')

    regress_mini_mtf()

    # try to get a non-existant table
    with pytest.raises(RuntimeError) as excinfo:
        pipeline.get_table("bogus")
    assert "never checkpointed" in str(excinfo.value)

    # try to get an existing table from a non-existant checkpoint
    with pytest.raises(RuntimeError) as excinfo:
        pipeline.get_table("households", checkpoint_name="bogus")
    assert "not in checkpoints" in str(excinfo.value)

    pipeline.close_pipeline()
    inject.clear_cache()
    close_handlers()


def test_mini_pipeline_run2():

    # the important thing here is that we should get
    # exactly the same results as for test_mini_pipeline_run
    # when we restart pipeline

    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')

    setup_dirs(configs_dir)

    inject_settings(configs_dir, households_sample_size=HOUSEHOLDS_SAMPLE_SIZE)

    # should be able to get this BEFORE pipeline is opened
    checkpoints_df = pipeline.get_checkpoints()
    prev_checkpoint_count = len(checkpoints_df.index)

    # print "checkpoints_df\n", checkpoints_df[['checkpoint_name']]
    assert prev_checkpoint_count == 8

    pipeline.open_pipeline('auto_ownership_simulate')

    regress_mini_auto()

    # try to run a model already in pipeline
    with pytest.raises(RuntimeError) as excinfo:
        pipeline.run_model('auto_ownership_simulate')
    assert "run model 'auto_ownership_simulate' more than once" in str(excinfo.value)

    # and these new ones
    pipeline.run_model('cdap_simulate')
    pipeline.run_model('mandatory_tour_frequency')

    regress_mini_mtf()

    # should be able to get this before pipeline is closed (from existing open store)
    checkpoints_df = pipeline.get_checkpoints()
    assert len(checkpoints_df.index) == prev_checkpoint_count

    # - write list of override_hh_ids to override_hh_ids.csv in data for use in next test
    num_hh_ids = 10
    hh_ids = pipeline.get_table("households").head(num_hh_ids).index.values
    hh_ids = pd.DataFrame({'household_id': hh_ids})

    data_dir = inject.get_injectable('data_dir')
    hh_ids.to_csv(os.path.join(data_dir, 'override_hh_ids.csv'), index=False, header=True)

    pipeline.close_pipeline()
    inject.clear_cache()
    close_handlers()


def test_mini_pipeline_run3():

    # test that hh_ids setting overrides household sampling

    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
    setup_dirs(configs_dir)
    inject_settings(configs_dir, hh_ids='override_hh_ids.csv')

    households = inject.get_table('households').to_frame()

    override_hh_ids = pd.read_csv(config.data_file_path('override_hh_ids.csv'))

    print("\noverride_hh_ids\n", override_hh_ids)

    print("\nhouseholds\n", households.index)

    assert households.shape[0] == override_hh_ids.shape[0]
    assert households.index.isin(override_hh_ids.household_id).all()

    inject.clear_cache()
    close_handlers()


def full_run(resume_after=None, chunk_size=0,
             households_sample_size=HOUSEHOLDS_SAMPLE_SIZE,
             trace_hh_id=None, trace_od=None, check_for_variability=None):

    configs_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'example', 'configs')

    setup_dirs(configs_dir)

    settings = inject_settings(
        configs_dir,
        households_sample_size=households_sample_size,
        chunk_size=chunk_size,
        trace_hh_id=trace_hh_id,
        trace_od=trace_od,
        check_for_variability=check_for_variability,
        use_shadow_pricing=False)  # shadow pricing breaks replicability when sample_size varies

    MODELS = settings['models']

    pipeline.run(models=MODELS, resume_after=resume_after)

    tours = pipeline.get_table('tours')
    tour_count = len(tours.index)

    return tour_count


def get_trace_csv(file_name):

    output_dir = os.path.join(os.path.dirname(__file__), 'output')

    df = pd.read_csv(os.path.join(output_dir, file_name))

    #        label    value_1    value_2    value_3    value_4
    # 0    tour_id        38         201         39         40
    # 1       mode  DRIVE_LOC  DRIVE_COM  DRIVE_LOC  DRIVE_LOC
    # 2  person_id    1888694    1888695    1888695    1888696
    # 3  tour_type       work   othmaint       work     school
    # 4   tour_num          1          1          1          1

    # transpose df and rename columns
    labels = df.label.values
    df = df.transpose()[1:]
    df.columns = labels

    return df


EXPECT_TOUR_COUNT = 345


def regress_tour_modes(tours_df):

    mode_cols = ['tour_mode', 'person_id', 'tour_type',
                 'tour_num', 'tour_category']

    tours_df = tours_df[tours_df.household_id == HH_ID]
    tours_df = tours_df.sort_values(by=['person_id', 'tour_category', 'tour_num'])

    print("mode_df\n", tours_df[mode_cols])

    """
     tour_id         tour_mode  person_id tour_type  tour_num  tour_category
    tour_id                                                                
    138094270            WALK    4761871  othdiscr         1          joint
    138094266  DRIVEALONEFREE    4761871    escort         1  non_mandatory
    138094284     SHARED2FREE    4761871  shopping         2  non_mandatory
    138094281  DRIVEALONEFREE    4761871  othmaint         3  non_mandatory
    138094314        WALK_LOC    4761872    social         1  non_mandatory
    138094340        WALK_LOC    4761873    school         1      mandatory
    138094342     SHARED3FREE    4761873  shopping         1  non_mandatory
    138094371     SHARED2FREE    4761874  shopping         1  non_mandatory
    138094440     SHARED2FREE    4761877    escort         1  non_mandatory
    138094441  DRIVEALONEFREE    4761877    escort         2  non_mandatory
    138094454        WALK_LOC    4761877  othdiscr         3  non_mandatory
    138094462     SHARED3FREE    4761878  business         1         atwork
    138094489        WALK_LOC    4761878      work         1      mandatory
    138094469  DRIVEALONEFREE    4761878    escort         1  non_mandatory
    138094487     SHARED2FREE    4761878  shopping         2  non_mandatory
    138094468            WALK    4761878    eatout         3  non_mandatory
    138094488            WALK    4761878    social         4  non_mandatory
    138094545            WALK    4761880  shopping         1  non_mandatory
    138094542            WALK    4761880  othmaint         2  non_mandatory
    138094576            WALK    4761881      work         1      mandatory
    138094577            WALK    4761881      work         2      mandatory
    """

    EXPECT_PERSON_IDS = [
        4761871,
        4761871,
        4761871,
        4761871,
        4761872,
        4761873,
        4761873,
        4761874,
        4761877,
        4761877,
        4761877,
        4761878,
        4761878,
        4761878,
        4761878,
        4761878,
        4761878,
        4761880,
        4761880,
        4761881,
        4761881,
        ]

    EXPECT_TOUR_TYPES = [
        'othdiscr',
        'escort',
        'shopping',
        'othmaint',
        'social',
        'school',
        'shopping',
        'shopping',
        'escort',
        'escort',
        'othdiscr',
        'business',
        'work',
        'escort',
        'shopping',
        'eatout',
        'social',
        'shopping',
        'othmaint',
        'work',
        'work',
        ]

    EXPECT_MODES = [
        'WALK',
        'DRIVEALONEFREE',
        'SHARED2FREE',
        'DRIVEALONEFREE',
        'WALK_LOC',
        'WALK_LOC',
        'SHARED3FREE',
        'SHARED2FREE',
        'SHARED2FREE',
        'DRIVEALONEFREE',
        'WALK_LOC',
        'SHARED3FREE',
        'WALK_LOC',
        'DRIVEALONEFREE',
        'SHARED2FREE',
        'WALK',
        'WALK',
        'WALK',
        'WALK',
        'WALK',
        'WALK',
        ]

    assert (tours_df.person_id.values == EXPECT_PERSON_IDS).all()
    assert (tours_df.tour_type.values == EXPECT_TOUR_TYPES).all()
    assert (tours_df.tour_mode.values == EXPECT_MODES).all()


def regress():

    persons_df = pipeline.get_table('persons')
    persons_df = persons_df[persons_df.household_id == HH_ID]
    print("persons_df\n", persons_df[['value_of_time', 'distance_to_work']])

    """
    persons_df
     person_id  value_of_time  distance_to_work
    person_id
    3249922        23.349532              0.62
    3249923        23.349532              0.62
    """

    tours_df = pipeline.get_table('tours')

    regress_tour_modes(tours_df)

    assert tours_df.shape[0] > 0
    assert not tours_df.tour_mode.isnull().any()

    trips_df = pipeline.get_table('trips')
    assert trips_df.shape[0] > 0
    assert not trips_df.purpose.isnull().any()
    assert not trips_df.depart.isnull().any()
    assert not trips_df.trip_mode.isnull().any()

    # should be at least two tours per trip
    assert trips_df.shape[0] >= 2*tours_df.shape[0]


def test_full_run1():

    if SKIP_FULL_RUN:
        return

    tour_count = full_run(trace_hh_id=HH_ID, check_for_variability=True,
                          households_sample_size=HOUSEHOLDS_SAMPLE_SIZE)

    print("tour_count", tour_count)

    assert(tour_count == EXPECT_TOUR_COUNT)

    regress()

    pipeline.close_pipeline()


def test_full_run2():

    # resume_after should successfully load tours table and replicate results

    if SKIP_FULL_RUN:
        return

    tour_count = full_run(resume_after='non_mandatory_tour_scheduling', trace_hh_id=HH_ID)

    assert(tour_count == EXPECT_TOUR_COUNT)

    regress()

    pipeline.close_pipeline()


def test_full_run3_with_chunks():

    # should get the same result with different chunk size

    if SKIP_FULL_RUN:
        return

    tour_count = full_run(trace_hh_id=HH_ID,
                          households_sample_size=HOUSEHOLDS_SAMPLE_SIZE,
                          chunk_size=500000)

    assert(tour_count == EXPECT_TOUR_COUNT)

    regress()

    pipeline.close_pipeline()


def test_full_run4_stability():

    # hh should get the same result with different sample size

    if SKIP_FULL_RUN:
        return

    tour_count = full_run(trace_hh_id=HH_ID,
                          households_sample_size=HOUSEHOLDS_SAMPLE_SIZE+10)

    regress()

    pipeline.close_pipeline()


def test_full_run5_singleton():

    # should wrk with only one hh

    if SKIP_FULL_RUN:
        return

    tour_count = full_run(trace_hh_id=HH_ID,
                          households_sample_size=1)

    regress()

    pipeline.close_pipeline()


if __name__ == "__main__":

    print("running test_full_run1")
    test_full_run1()
    # teardown_function(None)
