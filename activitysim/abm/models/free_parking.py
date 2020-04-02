# ActivitySim
# See full license in LICENSE.txt.

import logging

from activitysim.core import tracing
from activitysim.core import config
from activitysim.core import pipeline
from activitysim.core import simulate
from activitysim.core import inject

from .util import expressions
from .util import estimation

logger = logging.getLogger(__name__)


@inject.step()
def free_parking(
        persons_merged, persons, households,
        skim_dict, skim_stack,
        chunk_size, trace_hh_id, locutor):
    """

    """

    trace_label = 'free_parking'
    model_settings = config.read_model_settings('free_parking.yaml')

    choosers = persons_merged.to_frame()
    choosers = choosers[choosers.workplace_taz > -1]

    logger.info("Running %s with %d persons", trace_label, len(choosers))

    constants = config.get_model_constants(model_settings)

    # - preprocessor
    preprocessor_settings = model_settings.get('preprocessor', None)
    if preprocessor_settings:

        locals_d = {}
        if constants is not None:
            locals_d.update(constants)

        expressions.assign_columns(
            df=choosers,
            model_settings=preprocessor_settings,
            locals_dict=locals_d,
            trace_label=trace_label)


    model_spec = simulate.read_model_spec(model_settings=model_settings)
    coefficients_df = simulate.read_model_coefficients(model_settings)
    model_spec = simulate.eval_coefficients(model_spec, coefficients_df)

    nest_spec = config.get_logit_model_settings(model_settings)

    if estimation.manager.begin_estimation('free_parking'):
        estimation.manager.write_model_settings(model_settings, 'free_parking.yaml')
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
        trace_choice_name='free_parking_at_work',
        estimation_hook=estimation_hook)

    free_parking_alt = model_settings['FREE_PARKING_ALT']
    choices = (choices == free_parking_alt)

    if estimation.manager.estimating:
        estimation.manager.write_choices(choices)
        choices = estimation.manager.get_override_choices(choices)
        estimation.manager.end_estimation()

    persons = persons.to_frame()
    persons['free_parking_at_work'] = choices.reindex(persons.index).fillna(0).astype(bool)

    pipeline.replace_table("persons", persons)

    tracing.print_summary('free_parking', persons.free_parking_at_work, value_counts=True)

    if trace_hh_id:
        tracing.trace_df(persons,
                         label=trace_label,
                         warn_if_empty=True)
