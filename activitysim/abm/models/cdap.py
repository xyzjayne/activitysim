# ActivitySim
# See full license in LICENSE.txt.

from __future__ import (absolute_import, division, print_function, )
from future.standard_library import install_aliases
install_aliases()  # noqa: E402

import logging

import pandas as pd

from activitysim.core import simulate
from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import config
from activitysim.core import inject

from .util import cdap
from .util import expressions
from .util import estimation

logger = logging.getLogger(__name__)

@inject.step()
def cdap_simulate(persons_merged, persons, households,
                  chunk_size, trace_hh_id):
    """
    CDAP stands for Coordinated Daily Activity Pattern, which is a choice of
    high-level activity pattern for each person, in a coordinated way with other
    members of a person's household.

    Because Python requires vectorization of computation, there are some specialized
    routines in the cdap directory of activitysim for this purpose.  This module
    simply applies those utilities using the simulation framework.
    """

    trace_label = 'cdap'
    model_settings = config.read_model_settings('cdap.yaml')

    cdap_indiv_spec = simulate.read_model_spec(model_settings=model_settings, tag='INDIV_AND_HHSIZE1_SPEC')

    # Rules and coefficients for generating interaction specs for different household sizes
    cdap_interaction_coefficients = \
        pd.read_csv(config.config_file_path('cdap_interaction_coefficients.csv'), comment='#')

    """
    spec to compute/specify the relative proportions of each activity (M, N, H)
    that should be used to choose activities for additional household members not handled by CDAP
    This spec is handled much like an activitysim logit utility spec,
    EXCEPT that the values computed are relative proportions, not utilities
    (i.e. values are not exponentiated before being normalized to probabilities summing to 1.0)
    """
    cdap_fixed_relative_proportions = \
        simulate.read_model_spec(model_settings=model_settings, tag='FIXED_RELATIVE_PROPORTIONS_SPEC')

    persons_merged = persons_merged.to_frame()

    constants = config.get_model_constants(model_settings)

    cdap_interaction_coefficients = \
        cdap.preprocess_interaction_coefficients(cdap_interaction_coefficients)

    # specs are built just-in-time on demand and cached as injectables
    # prebuilding here allows us to write them to the output directory
    # (also when multiprocessing locutor might not see all household sizes)
    logger.info("Pre-building cdap specs")
    for hhsize in range(2, cdap.MAX_HHSIZE + 1):
        spec = cdap.build_cdap_spec(cdap_interaction_coefficients, hhsize, cache=True)
        if inject.get_injectable('locutor', False):
            spec.to_csv(config.output_file_path('cdap_spec_%s.csv' % hhsize), index=True)

    if estimation.manager.begin_estimation('cdap'):
        estimation.manager.write_model_settings(model_settings, 'cdap.yaml')
        estimation.manager.write_spec(model_settings, tag='INDIV_AND_HHSIZE1_SPEC')
        estimation.manager.write_spec(model_settings=model_settings, tag='FIXED_RELATIVE_PROPORTIONS_SPEC')
        estimation.manager.write_table(cdap_interaction_coefficients, 'interaction_coefficients', index=False, append=False)
        estimation.manager.write_choosers(persons_merged)
        estimation_hook = estimation.write_hook
        for hhsize in range(2, cdap.MAX_HHSIZE + 1):
            spec = cdap.get_cached_spec(hhsize)
            estimation.manager.write_table(spec, 'spec_%s' % hhsize, append=False)
    else:
        estimation_hook = None

    logger.info("Running cdap_simulate with %d persons", len(persons_merged.index))

    choices = cdap.run_cdap(
        persons=persons_merged,
        cdap_indiv_spec=cdap_indiv_spec,
        cdap_interaction_coefficients=cdap_interaction_coefficients,
        cdap_fixed_relative_proportions=cdap_fixed_relative_proportions,
        locals_d=constants,
        chunk_size=chunk_size,
        trace_hh_id=trace_hh_id,
        trace_label=trace_label,
        estimation_hook=estimation_hook)

    if estimation.manager.estimating:
        estimation.manager.write_choices(choices)
        choices = estimation.manager.get_override_choices(choices)
        estimation.manager.end_estimation()

    # - assign results to persons table and annotate
    persons = persons.to_frame()

    choices = choices.reindex(persons.index)
    persons['cdap_activity'] = choices

    expressions.assign_columns(
        df=persons,
        model_settings=model_settings.get('annotate_persons'),
        trace_label=tracing.extend_trace_label(trace_label, 'annotate_persons'))

    pipeline.replace_table("persons", persons)

    # - annotate households table
    households = households.to_frame()
    expressions.assign_columns(
        df=households,
        model_settings=model_settings.get('annotate_households'),
        trace_label=tracing.extend_trace_label(trace_label, 'annotate_households'))
    pipeline.replace_table("households", households)

    tracing.print_summary('cdap_activity', persons.cdap_activity, value_counts=True)
    logger.info("cdap crosstabs:\n%s" %
                pd.crosstab(persons.ptype, persons.cdap_activity, margins=True))

    if trace_hh_id:

        tracing.trace_df(inject.get_table('persons_merged').to_frame(),
                         label="cdap",
                         columns=['ptype', 'cdap_rank', 'cdap_activity'],
                         warn_if_empty=True)
