# ActivitySim
# See full license in LICENSE.txt.

# from __future__ import (absolute_import, division, print_function, )
# from future.standard_library import install_aliases
# install_aliases()  # noqa: E402

import logging

from activitysim.core import simulate
from activitysim.core import tracing
from activitysim.core import pipeline
from activitysim.core import config
from activitysim.core import inject

from .util import estimation


logger = logging.getLogger(__name__)


@inject.step()
def auto_ownership_simulate(households,
                            households_merged,
                            chunk_size,
                            trace_hh_id):
    """
    Auto ownership is a standard model which predicts how many cars a household
    with given characteristics owns
    """
    trace_label = 'auto_ownership_simulate'
    model_settings = config.read_model_settings('auto_ownership.yaml')

    logger.info("Running %s with %d households", trace_label, len(households_merged))

    model_spec = simulate.read_model_spec(model_settings=model_settings)

    coefficients_df = simulate.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(model_spec, coefficients_df)

    nest_spec = config.get_logit_model_settings(model_settings)
    constants = config.get_model_constants(model_settings)

    choosers = households_merged.to_frame()

    if estimation.manager.begin_estimation('auto_ownership'):
        estimation.manager.write_model_settings(model_settings, 'auto_ownership.yaml')
        estimation.manager.write_spec(model_settings)
        estimation.manager.write_coefficients(coefficients_df)
        estimation.manager.write_choosers(choosers)
        estimation_hook = estimation.write_hook
    else:
        estimation_hook = None

    choices = simulate.simple_simulate(
        choosers=choosers,
        spec=model_spec,
        nest_spec=nest_spec,
        locals_d=constants,
        chunk_size=chunk_size,
        trace_label=trace_label,
        trace_choice_name='auto_ownership',
        estimation_hook=estimation_hook)

    if estimation.manager.estimating:
        estimation.manager.write_choices(choices)
        choices = estimation.manager.get_override_choices(choices)

        estimation.manager.end_estimation()

    households = households.to_frame()

    # no need to reindex as we used all households
    households['auto_ownership'] = choices

    pipeline.replace_table("households", households)

    tracing.print_summary('auto_ownership', households.auto_ownership, value_counts=True)

    if trace_hh_id:
        tracing.trace_df(households,
                         label='auto_ownership',
                         warn_if_empty=True)