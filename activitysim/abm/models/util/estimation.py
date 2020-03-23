# ActivitySim
# See full license in LICENSE.txt.

import os
import shutil

import logging

import yaml

import pandas as pd

from activitysim.core import config
from activitysim.core import inject

logger = logging.getLogger(__name__)


def estimation_model_name():
    return inject.get_injectable('estimation_model_name', None)


def begin_estimation(model_name):
    """
    register injectible with current model name
    (so we don't have to pass it everywhere or rely on trace_label)

    Delete estimation files so we can concat results for all chunks/segments in a model run
    (but without dregs from the last run.)

    Returns
    -------
    Nothing
    """

    inject.add_injectable('estimation_model_name', model_name)

    # ensure the output data directory exists
    output_dir = estimation_data_directory()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # make directory if needed

    # delete estimation files
    file_type = ('csv', 'yaml')
    for file_name in os.listdir(output_dir):
        if file_name.startswith(model_name) and file_name.endswith(file_type):
            file_path = os.path.join(output_dir, file_name)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(e)


def end_estimation():

    inject.add_injectable('estimation_model_name', None)

    #bug
    exit()


def estimating(model_name=None):

    if model_name is not None:

        if estimation_model_name() is not None:
            # shouldn't already be estimating
            assert False

        if model_name == config.setting('model_estimation', False):
            begin_estimation(model_name)

    return estimation_model_name() is not None


def estimation_data_directory():

    assert estimation_model_name() is not None

    data_bundle_dir = config.output_file_path('estimation_data_bundle')

    return os.path.join(data_bundle_dir, estimation_model_name())


def estimation_file_path(table_name, file_type=None):

    model_name = estimation_model_name()
    file_name = "%s_%s.%s" % (model_name, table_name, file_type) if file_type else "%s_%s" % (model_name, table_name)

    output_dir = estimation_data_directory()
    return os.path.join(output_dir, file_name)


def write_table(df, table_name, index=True, append=True):

    assert estimation_model_name() is not None

    file_path = estimation_file_path(table_name, 'csv')

    file_exists = os.path.isfile(file_path)

    if file_exists and not append:
        raise RuntimeError("write_table %s append=False and file exists: %s" % (table_name, file_path))

    df.to_csv(file_path, mode='a', index=index, header=(not file_exists))

    print('estimate.write_table', file_path)


def write_dict(d, dict_name):

    assert estimation_model_name() is not None

    file_path = estimation_file_path(dict_name, 'yaml')

    # we don't know how to concat, and afraid to overwrite
    assert not os.path.isfile(file_path)

    with open(file_path, 'w') as f:
        # write ordered dict as array
        yaml.dump(d, f)

    print('estimate.write_dict', file_path)


def write_coefficients(coefficients_df):

    write_table(coefficients_df, 'coefficients', index=True, append=False)


def write_choosers(choosers_df):

    write_table(choosers_df, 'choosers', index=True, append=True)


def write_alternatives(alternatives_df):

    write_table(alternatives_df, 'alternatives', index=True, append=True)


def write_choices(choices):
    # rename first column
    write_table(choices.to_frame(name='choices'), 'choices', index=True, append=True)


def write_constants(constants):

    write_dict(constants, 'model_constants')


def write_nest_spec(nest_spec):

    write_dict(nest_spec, 'nest_spec')


def base_settings_file_path(file_name):
    """

    FIXME - should be in configs

    Parameters
    ----------
    file_name

    Returns
    -------
        path to base settings file or None if not found
    """

    if not file_name.lower().endswith('.yaml'):
        file_name = '%s.yaml' % (file_name, )

    configs_dir = inject.get_injectable('configs_dir')

    if isinstance(configs_dir, str):
        configs_dir = [configs_dir]

    assert isinstance(configs_dir, list)

    for dir in configs_dir:
        file_path = os.path.join(dir, file_name)
        if os.path.exists(file_path):
            return file_path

    raise RuntimeError("base_settings_file %s not found" % file_name)


def copy_model_settings(settings_file_name, tag='model_settings'):

    input_path = base_settings_file_path(settings_file_name)

    output_path = estimation_file_path(tag, 'yaml')

    shutil.copy(input_path, output_path)


def write_model_settings(model_settings, settings_file_name):

    copy_model_settings(settings_file_name)
    if 'inherit_settings' in model_settings:
        write_dict(model_settings, 'inherited_model_settings')


def write_spec(model_settings, tag='SPEC'):
    # FIXME  should also copy like write_model_settings (when possible?) to capture comment lines in csv?

    # estimation.write_spec(simulate.read_model_spec(file_name=spec_file_name, tag='sample_spec')
    spec_file_name = model_settings[tag]

    # spec_df = simulate.read_model_spec(file_name=spec_file_name)
    # write_table(spec_df, tag, append=False)

    # if not spec_file_name.lower().endswith('.yaml'):
    #     file_name = '%s.yaml' % (spec_file_name, )

    input_path = config.config_file_path(spec_file_name)
    output_path = estimation_file_path(table_name=tag, file_type='csv')

    shutil.copy(input_path, output_path)


def write_hook(df, table_name):
    if table_name == 'expression_values':
        # mergesort is the only stable sort, and we want the expressions to appear in original df column order
        index_name = df.index.name
        df = pd.melt(df.reset_index(), id_vars=[index_name]).sort_values(by=index_name, kind='mergesort')
        write_table(df, table_name, index=False)


    elif table_name == 'interaction_expression_values':

        # mergesort is the only stable sort, and we want the expressions to appear in original df column order
        index_name = df.index.name
        alt_id_name = 'alt_dest'
        df = pd.melt(df.reset_index(), id_vars=[index_name, alt_id_name])\
            .sort_values(by=index_name, kind='mergesort')\
            .rename(columns={'variable': 'expression'})

        # person_id,alt_dest,expression,value
        # 31153,1,"@_DIST.clip(0,1)",1.0
        # 31153,2,"@_DIST.clip(0,1)",1.0
        # 31153,3,"@_DIST.clip(0,1)",1.0
        # 31153,4,"@_DIST.clip(0,1)",1.0

        df = df.set_index([index_name, 'expression', alt_id_name]).unstack(2)
        df.columns = df.columns.droplevel(0)
        df = df.reset_index(1)

        # person_id,expression,1,2,3,4,5,6
        # 31153,"@(_DIST-1).clip(0,1)",0.75,0.46,0.27,0.63,0.48,0.23
        # 31153,@(_DIST-15.0).clip(0),0.0,0.0,0.0,0.0,0.0,0.0
        # 31153,"@(_DIST-2).clip(0,3)",0.0,0.0,0.0,0.0,0.0,0.0

        write_table(df, table_name, index=True)

    else:
        write_table(df, table_name)