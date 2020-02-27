# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

from future.utils import iteritems

import logging

import pandas as pd

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import simulate
from activitysim.core import inject
from activitysim.core.mem import force_garbage_collect

from activitysim.core.interaction_sample_simulate import interaction_sample_simulate
from activitysim.core.interaction_sample import interaction_sample

from .util import expressions
from .util import logsums as logsum

from activitysim.abm.tables import shadow_pricing

"""
The school/workplace location model predicts the zones in which various people will
work or attend school.

For locations choices like workplace and school location, we have existing data about the actual
number of workers or students in the various destination zones, and we naturally want the results
of location choice to yield distributions the match these observed distributions as closely as
possible. To achieve this, we use start with size tables with the observed populations by zone
and segment (e.g. number of university, highschool, and gradeschool students in each zone) and
use those populations as attractors (positive utilities) so that high population zones will,
all things being equal, receive more choices. (For instance, we want university-goers to choose
school locations with in zones with university enrollments.)

But since the choice algorithm can result in aggregate distributions of choices (modeled_size)
that don't match observed (predicted_size) counts. The shadow pricing algorithm attempts to
correct these misalignments, by iteratively running the choice model, comparing the modeled_size
of the zones segments to the  predicted size, and computing a shadow_price coefficient that is
applied to the size term to boost or attenuate its influence. This iterative process can be
configures to continue until a specified closeness of fit is achieved, or a maximum number of
iterations has occurred. Since the iterative process can be expensive, a facility is provided
to save the computed shadow prices after every iteration, and to load pre-computed shadow prices
on subsequent runs (warm start) to cut down on runtimes.

Since every individual (always person for now) belongs to at most one segment, each segment
(e.g. 'university', 'highschool' , 'gradeschool' for the 'school' location model) is handled
separately and sequentially withing each shadow-price iteration.

The core algorithm has 3 parts:

Because logsum calculations are expensive, rather than computing logsums for all destination
alternatives, we first build a sample of alternatives using simplified (no-logsum) utilities,
and compute logsums only for that sample, and finally chose from among the sampled alternatives.

* run_location_sample - Build a sample destination alternatives using simplified choice criteria
* run_location_logsums - Compute logsums for travel to those alternatives
* run_location_simulate - Rerun the choice model using the logsums to make a final location choice

With shadow pricing, and iterative treatment of each segment, the structure of the code is:

::

    repeat
        for each segment
            run_location_sample
            run_location_logsums
            run_location_simulate
    until convergence

The location choice is added to a column in choosers
The column name is specified in the model_settings by DEST_CHOICE_COLUMN_NAME.

Additionally, providing DEST_CHOICE_LOGSUM_COLUMN_NAME will add a column with the logsum of the
utilities of the sampled location alternatives (the logarithm of the sum of the exponentiated
utilities of all the alternatives.) This optional value is not used in the existing models
(at least not in MTCTM1) but it is a capability provided by CTRAMP, and it apparently of use
calibrating the model.
"""

logger = logging.getLogger(__name__)


def spec_for_segment(model_spec, segment_name):
    """
    Select spec for specified segment from omnibus spec containing columns for each segment

    Parameters
    ----------
    model_spec : pandas.DataFrame
        omnibus spec file with expressions in index and one column per segment
    segment_name : str
        segment_name that is also column name in model_spec

    Returns
    -------
    pandas.dataframe
        canonical spec file with expressions in index and single column with utility coefficients
    """

    spec = model_spec[[segment_name]]

    # drop spec rows with zero coefficients since they won't have any effect (0 marginal utility)
    zero_rows = (spec == 0).all(axis=1)
    if zero_rows.any():
        logger.debug("dropping %s all-zero rows from spec" % (zero_rows.sum(),))
        spec = spec.loc[~zero_rows]

    return spec


def run_location_sample(
        segment_name,
        persons_merged,
        skim_dict,
        dest_size_terms,
        model_settings,
        chunk_size, trace_label):
    """
    select a sample of alternative locations.

    Logsum calculations are expensive, so we build a table of persons * all zones
    and then select a sample subset of potential locations

    The sample subset is generated by making multiple choices (<sample_size> number of choices)
    which results in sample containing up to <sample_size> choices for each choose (e.g. person)
    and a pick_count indicating how many times that choice was selected for that chooser.)

    person_id,  dest_TAZ, rand,            pick_count
    23750,      14,       0.565502716034,  4
    23750,      16,       0.711135838871,  6
    ...
    23751,      12,       0.408038878552,  1
    23751,      14,       0.972732479292,  2
    """
    assert not persons_merged.empty

    model_spec = simulate.read_model_spec(file_name=model_settings['SAMPLE_SPEC'])

    # FIXME - MEMORY HACK - only include columns actually used in spec
    chooser_columns = model_settings['SIMULATE_CHOOSER_COLUMNS']
    choosers = persons_merged[chooser_columns]

    alternatives = dest_size_terms

    sample_size = model_settings['SAMPLE_SIZE']
    alt_dest_col_name = model_settings['ALT_DEST_COL_NAME']

    logger.info("Running %s with %d persons" % (trace_label, len(choosers.index)))

    # create wrapper with keys for this lookup - in this case there is a TAZ in the choosers
    # and a TAZ in the alternatives which get merged during interaction
    # (logit.interaction_dataset suffixes duplicate chooser column with '_chooser')
    # the skims will be available under the name "skims" for any @ expressions
    skims = skim_dict.wrap('TAZ_chooser', 'TAZ')

    locals_d = {
        'skims': skims,
        'segment_size': segment_name
    }
    constants = config.get_model_constants(model_settings)
    if constants is not None:
        locals_d.update(constants)

    choices = interaction_sample(
        choosers,
        alternatives,
        sample_size=sample_size,
        alt_col_name=alt_dest_col_name,
        spec=spec_for_segment(model_spec, segment_name),
        skims=skims,
        locals_d=locals_d,
        chunk_size=chunk_size,
        trace_label=trace_label)

    return choices


def run_location_logsums(
        segment_name,
        persons_merged_df,
        skim_dict, skim_stack,
        location_sample_df,
        model_settings,
        chunk_size, trace_hh_id, trace_label):
    """
    add logsum column to existing location_sample table

    logsum is calculated by running the mode_choice model for each sample (person, dest_taz) pair
    in location_sample, and computing the logsum of all the utilities

    +-----------+--------------+----------------+------------+----------------+
    | PERID     | dest_TAZ     | rand           | pick_count | logsum (added) |
    +===========+==============+================+============+================+
    | 23750     |  14          | 0.565502716034 | 4          |  1.85659498857 |
    +-----------+--------------+----------------+------------+----------------+
    + 23750     | 16           | 0.711135838871 | 6          | 1.92315598631  |
    +-----------+--------------+----------------+------------+----------------+
    + ...       |              |                |            |                |
    +-----------+--------------+----------------+------------+----------------+
    | 23751     | 12           | 0.408038878552 | 1          | 2.40612135416  |
    +-----------+--------------+----------------+------------+----------------+
    | 23751     | 14           | 0.972732479292 | 2          |  1.44009018355 |
    +-----------+--------------+----------------+------------+----------------+
    """

    assert not location_sample_df.empty

    logsum_settings = config.read_model_settings(model_settings['LOGSUM_SETTINGS'])

    # FIXME - MEMORY HACK - only include columns actually used in spec
    persons_merged_df = \
        logsum.filter_chooser_columns(persons_merged_df, logsum_settings, model_settings)

    logger.info("Running %s with %s rows" % (trace_label, len(location_sample_df.index)))

    choosers = location_sample_df.join(persons_merged_df, how='left')

    tour_purpose = model_settings['LOGSUM_TOUR_PURPOSE']
    if isinstance(tour_purpose, dict):
        tour_purpose = tour_purpose[segment_name]

    logsums = logsum.compute_logsums(
        choosers,
        tour_purpose,
        logsum_settings, model_settings,
        skim_dict, skim_stack,
        chunk_size, trace_hh_id,
        trace_label)

    # "add_column series should have an index matching the table to which it is being added"
    # when the index has duplicates, however, in the special case that the series index exactly
    # matches the table index, then the series value order is preserved
    # logsums now does, since workplace_location_sample was on left side of merge de-dup merge
    location_sample_df['mode_choice_logsum'] = logsums

    return location_sample_df


def run_location_simulate(
        segment_name,
        persons_merged,
        location_sample_df,
        skim_dict,
        dest_size_terms,
        want_logsums,
        model_settings,
        chunk_size, trace_label):
    """
    run location model on location_sample annotated with mode_choice logsum
    to select a dest zone from sample alternatives

    Returns
    -------
    choices : pandas.DataFrame indexed by persons_merged_df.index
        choice : location choices (zone ids)
        logsum : float logsum of choice utilities across alternatives

    logsums optional & only returned if DEST_CHOICE_LOGSUM_COLUMN_NAME specified in model_settings
    """
    assert not persons_merged.empty

    model_spec = simulate.read_model_spec(file_name=model_settings['SPEC'])

    # FIXME - MEMORY HACK - only include columns actually used in spec
    chooser_columns = model_settings['SIMULATE_CHOOSER_COLUMNS']
    choosers = persons_merged[chooser_columns]

    alt_dest_col_name = model_settings['ALT_DEST_COL_NAME']

    # alternatives are pre-sampled and annotated with logsums and pick_count
    # but we have to merge additional alt columns into alt sample list
    alternatives = \
        pd.merge(location_sample_df, dest_size_terms,
                 left_on=alt_dest_col_name, right_index=True, how="left")

    logger.info("Running %s with %d persons" % (trace_label, len(choosers)))

    # create wrapper with keys for this lookup - in this case there is a TAZ in the choosers
    # and a TAZ in the alternatives which get merged during interaction
    # the skims will be available under the name "skims" for any @ expressions
    skims = skim_dict.wrap("TAZ_chooser", alt_dest_col_name)

    locals_d = {
        'skims': skims,
        'segment_size': segment_name
    }
    constants = config.get_model_constants(model_settings)
    if constants is not None:
        locals_d.update(constants)

    choices = interaction_sample_simulate(
        choosers,
        alternatives,
        spec=spec_for_segment(model_spec, segment_name),
        choice_column=alt_dest_col_name,
        want_logsums=want_logsums,
        skims=skims,
        locals_d=locals_d,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name=model_settings['DEST_CHOICE_COLUMN_NAME'])

    if not want_logsums:
        # for consistency, always return a dataframe with canonical column name
        assert isinstance(choices, pd.Series)
        choices = choices.to_frame('choice')

    assert isinstance(choices, pd.DataFrame)

    return choices


def run_location_choice(
        persons_merged_df,
        skim_dict, skim_stack,
        shadow_price_calculator,
        want_logsums,
        want_sample_table,
        model_settings,
        chunk_size, trace_hh_id, trace_label
        ):
    """
    Run the three-part location choice algorithm to generate a location choice for each chooser

    Handle the various segments separately and in turn for simplicity of expression files

    Parameters
    ----------
    persons_merged_df : pandas.DataFrame
        persons table merged with households and land_use
    skim_dict : skim.SkimDict
    skim_stack : skim.SkimStack
    shadow_price_calculator : ShadowPriceCalculator
        to get size terms
    want_logsums : boolean
    want_sample_table : boolean
    model_settings : dict
    chunk_size : int
    trace_hh_id : int
    trace_label : str

    Returns
    -------
    choices : pandas.DataFrame indexed by persons_merged_df.index
        'choice' : location choices (zone ids)
        'logsum' : float logsum of choice utilities across alternatives

    logsums optional & only returned if DEST_CHOICE_LOGSUM_COLUMN_NAME specified in model_settings
    """

    chooser_segment_column = model_settings['CHOOSER_SEGMENT_COLUMN_NAME']

    # maps segment names to compact (integer) ids
    segment_ids = model_settings['SEGMENT_IDS']

    choices_list = []
    sample_list = []
    for segment_name, segment_id in iteritems(segment_ids):

        choosers = persons_merged_df[persons_merged_df[chooser_segment_column] == segment_id]

        if choosers.shape[0] == 0:
            logger.info("%s skipping segment %s: no choosers", trace_label, segment_name)
            continue

        # size_term and shadow price adjustment - one row per zone
        dest_size_terms = shadow_price_calculator.dest_size_terms(segment_name)

        # - location_sample
        location_sample_df = \
            run_location_sample(
                segment_name,
                choosers,
                skim_dict,
                dest_size_terms,
                model_settings,
                chunk_size,
                tracing.extend_trace_label(trace_label, 'sample.%s' % segment_name))

        # - location_logsums
        location_sample_df = \
            run_location_logsums(
                segment_name,
                choosers,
                skim_dict, skim_stack,
                location_sample_df,
                model_settings,
                chunk_size,
                trace_hh_id,
                tracing.extend_trace_label(trace_label, 'logsums.%s' % segment_name))

        # - location_simulate
        choices = \
            run_location_simulate(
                segment_name,
                choosers,
                location_sample_df,
                skim_dict,
                dest_size_terms,
                want_logsums,
                model_settings,
                chunk_size,
                tracing.extend_trace_label(trace_label, 'simulate.%s' % segment_name))

        choices_list.append(choices)

        if want_sample_table:
            # FIXME - sample_table
            location_sample_df.set_index(model_settings['ALT_DEST_COL_NAME'],
                                         append=True, inplace=True)
            sample_list.append(location_sample_df)

        # FIXME - want to do this here?
        del location_sample_df
        force_garbage_collect()

    if len(choices_list) > 0:
        choices_df = pd.concat(choices_list)
    else:
        # this will only happen with small samples (e.g. singleton) with no (e.g.) school segs
        logger.warning("%s no choices", trace_label)
        choices_df = pd.DataFrame(columns=['choice', 'logsum'])

    if len(sample_list) > 0:
        save_sample_df = pd.concat(sample_list)
    else:
        # this could happen either with small samples as above, or if no saved sample desired
        save_sample_df = None

    return choices_df, save_sample_df


def iterate_location_choice(
        model_settings,
        persons_merged, persons, households,
        skim_dict, skim_stack,
        chunk_size, trace_hh_id, locutor,
        trace_label):
    """
    iterate run_location_choice updating shadow pricing until convergence criteria satisfied
    or max_iterations reached.

    (If use_shadow_pricing not enabled, then just iterate once)

    Parameters
    ----------
    model_settings : dict
    persons_merged : injected table
    persons : injected table
    skim_dict : skim.SkimDict
    skim_stack : skim.SkimStack
    chunk_size : int
    trace_hh_id : int
    locutor : bool
        whether this process is the privileged logger of shadow_pricing when multiprocessing
    trace_label : str

    Returns
    -------
    adds choice column model_settings['DEST_CHOICE_COLUMN_NAME']
    adds logsum column model_settings['DEST_CHOICE_LOGSUM_COLUMN_NAME']- if provided
    adds annotations to persons table
    """

    # column containing segment id
    chooser_segment_column = model_settings['CHOOSER_SEGMENT_COLUMN_NAME']

    # boolean to filter out persons not needing location modeling (e.g. is_worker, is_student)
    chooser_filter_column = model_settings['CHOOSER_FILTER_COLUMN_NAME']

    dest_choice_column_name = model_settings['DEST_CHOICE_COLUMN_NAME']
    logsum_column_name = model_settings.get('DEST_CHOICE_LOGSUM_COLUMN_NAME')
    sample_table_name = model_settings.get('DEST_CHOICE_SAMPLE_TABLE_NAME')

    persons_merged_df = persons_merged.to_frame()

    persons_merged_df = persons_merged_df[persons_merged[chooser_filter_column]]

    spc = shadow_pricing.load_shadow_price_calculator(model_settings)
    max_iterations = spc.max_iterations

    logging.debug("%s max_iterations: %s" % (trace_label, max_iterations))

    for iteration in range(1, max_iterations + 1):

        if spc.use_shadow_pricing and iteration > 1:
            spc.update_shadow_prices()

        choices_df, save_sample_df = run_location_choice(
            persons_merged_df,
            skim_dict, skim_stack,
            shadow_price_calculator=spc,
            want_logsums=logsum_column_name is not None,
            want_sample_table=sample_table_name is not None,
            model_settings=model_settings,
            chunk_size=chunk_size,
            trace_hh_id=trace_hh_id,
            trace_label=tracing.extend_trace_label(trace_label, 'i%s' % iteration))

        # choices_df is a pandas DataFrame with columns 'choice' and (optionally) 'logsum'
        if choices_df is None:
            break

        spc.set_choices(
            choices=choices_df['choice'],
            segment_ids=persons_merged_df[chooser_segment_column].reindex(choices_df.index))

        if locutor:
            spc.write_trace_files(iteration)

        if spc.use_shadow_pricing and spc.check_fit(iteration):
            logging.info("%s converged after iteration %s" % (trace_label, iteration,))
            break

    # - shadow price table
    if locutor:
        if spc.use_shadow_pricing and 'SHADOW_PRICE_TABLE' in model_settings:
            inject.add_table(model_settings['SHADOW_PRICE_TABLE'], spc.shadow_prices)
        if 'MODELED_SIZE_TABLE' in model_settings:
            inject.add_table(model_settings['MODELED_SIZE_TABLE'], spc.modeled_size)

    persons_df = persons.to_frame()

    # add the choice values to the dest_choice_column in persons dataframe
    # We only chose school locations for the subset of persons who go to school
    # so we backfill the empty choices with -1 to code as no school location
    # names for location choice and (optional) logsums columns
    NO_DEST_TAZ = -1
    persons_df[dest_choice_column_name] = \
        choices_df['choice'].reindex(persons_df.index).fillna(NO_DEST_TAZ).astype(int)

    # add the dest_choice_logsum column to persons dataframe
    if logsum_column_name:
        persons_df[logsum_column_name] = \
            choices_df['logsum'].reindex(persons_df.index).astype('float')

    if save_sample_df is not None:
        # might be None for tiny samples even if sample_table_name was specified
        assert len(save_sample_df.index.get_level_values(0).unique()) == len(choices_df)
        # lest they try to put school and workplace samples into the same table
        if pipeline.is_table(sample_table_name):
            raise RuntimeError("dest choice sample table %s already exists" % sample_table_name)
        pipeline.extend_table(sample_table_name, save_sample_df)

    # - annotate persons table
    if 'annotate_persons' in model_settings:
        expressions.assign_columns(
            df=persons_df,
            model_settings=model_settings.get('annotate_persons'),
            trace_label=tracing.extend_trace_label(trace_label, 'annotate_persons'))

        pipeline.replace_table("persons", persons_df)

        if trace_hh_id:
            tracing.trace_df(persons_df,
                             label=trace_label,
                             warn_if_empty=True)

    # - annotate households table
    if 'annotate_households' in model_settings:

        households_df = households.to_frame()
        expressions.assign_columns(
            df=households_df,
            model_settings=model_settings.get('annotate_households'),
            trace_label=tracing.extend_trace_label(trace_label, 'annotate_households'))
        pipeline.replace_table("households", households_df)

        if trace_hh_id:
            tracing.trace_df(households_df,
                             label=trace_label,
                             warn_if_empty=True)

    tracing.print_summary(dest_choice_column_name, choices_df['choice'], value_counts=True)
    if logsum_column_name:
        tracing.print_summary(logsum_column_name, choices_df['logsum'], value_counts=True)

    return persons_df


@inject.step()
def workplace_location(
        persons_merged, persons, households,
        skim_dict, skim_stack,
        chunk_size, trace_hh_id, locutor):
    """
    workplace location choice model

    iterate_location_choice adds location choice column and annotations to persons table
    """

    trace_label = 'workplace_location'
    model_settings = config.read_model_settings('workplace_location.yaml')

    iterate_location_choice(
        model_settings,
        persons_merged, persons, households,
        skim_dict, skim_stack,
        chunk_size, trace_hh_id, locutor, trace_label
    )


@inject.step()
def school_location(
        persons_merged, persons, households,
        skim_dict, skim_stack,
        chunk_size, trace_hh_id, locutor
        ):
    """
    School location choice model

    iterate_location_choice adds location choice column and annotations to persons table
    """

    trace_label = 'school_location'
    model_settings = config.read_model_settings('school_location.yaml')

    iterate_location_choice(
        model_settings,
        persons_merged, persons, households,
        skim_dict, skim_stack,
        chunk_size, trace_hh_id, locutor, trace_label
    )