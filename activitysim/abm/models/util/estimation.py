# ActivitySim
# See full license in LICENSE.txt.

import os
import shutil

import logging

import yaml

import pandas as pd

from activitysim.core import config
from activitysim.core import inject
from activitysim.core.util import reindex

logger = logging.getLogger(__name__)


class EstimationManager(object):

    def __init__(self):

        self.model_name = None
        self.estimating = False
        self.settings = None

    def begin_estimation(self, model_name):
        """
        begin estimating of model_name is specified as model to estimate, otherwise return False

        Parameters
        ----------
        model_name

        Returns
        -------

        """

        # load estimation settings file
        if self.settings is None:
            self.settings = config.read_model_settings('estimation.yaml')

        # shouldn't already be estimating
        assert not self.estimating, \
            "Cant begin estimating %s - already estimating %s" % (model_name, self.model_name)

        if model_name != self.settings.get('model', None):
            return False

        self.model_name = model_name
        self.estimating = True

        # ensure there are choice override settings for this model
        assert self.model_name in self.settings, \
            "Choice override settings for model %s not fund in estimatin settings file."
        self.choice_settings = self.settings[self.model_name]
        assert 'file_name' in self.choice_settings
        assert 'column_name' in self.choice_settings


        # ensure the output data directory exists
        output_dir = self.data_directory()
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

        return True

    def end_estimation(self):

        self.model_name = None
        self.estimating = False
        #bug
        exit()

    def data_directory(self):

        # shouldn't be asking for this if not estimating
        assert self.model_name is not None

        data_bundle_dir = config.output_file_path('estimation_data_bundle')

        return os.path.join(data_bundle_dir, self.model_name)

    def file_path(self, table_name, file_type=None):

        # shouldn't be asking for this if not estimating
        assert self.model_name is not None

        file_name = "%s_%s.%s" % (self.model_name, table_name, file_type) if file_type else "%s_%s" % (model_name, table_name)

        return os.path.join(self.data_directory(), file_name)

    def write_table(self, df, table_name, index=True, append=True):

        assert self.estimating
        assert self.model_name is not None

        file_path = self.file_path(table_name, 'csv')

        file_exists = os.path.isfile(file_path)

        if file_exists and not append:
            raise RuntimeError("write_table %s append=False and file exists: %s" % (table_name, file_path))

        df.to_csv(file_path, mode='a', index=index, header=(not file_exists))

        print('estimate.write_table', file_path)

    def write_dict(self, d, dict_name):

        assert self.estimating
        assert self.model_name is not None

        file_path = self.file_path(dict_name, 'yaml')

        # we don't know how to concat, and afraid to overwrite
        assert not os.path.isfile(file_path)

        with open(file_path, 'w') as f:
            # write ordered dict as array
            yaml.dump(d, f)

        logger.info("estimate.write_dict: %s" % file_path)


    def write_coefficients(self, coefficients_df):
        assert self.estimating
        self.write_table(coefficients_df, 'coefficients', index=True, append=False)

    def write_choosers(self, choosers_df):
        self.write_table(choosers_df, 'choosers', index=True, append=True)


    def write_alternatives(self, alternatives_df):
        self.write_table(alternatives_df, 'alternatives', index=True, append=True)


    def write_choices(self, choices):
        # rename first column
        self.write_table(choices.to_frame(name='choices'), 'choices', index=True, append=True)


    def write_constants(self, constants):
        self.write_dict(self, constants, 'model_constants')


    def write_nest_spec(self, nest_spec):
        self.write_dict(self, nest_spec, 'nest_spec')

    def copy_model_settings(self, settings_file_name, tag='model_settings'):

        input_path = config.base_settings_file_path(settings_file_name)

        output_path = self.file_path(tag, 'yaml')

        shutil.copy(input_path, output_path)


    def write_model_settings(self, model_settings, settings_file_name):

        self.copy_model_settings(settings_file_name)
        if 'inherit_settings' in model_settings:
            self.write_dict(model_settings, 'inherited_model_settings')


    def write_spec(self, model_settings, tag='SPEC'):
        # FIXME  should also copy like write_model_settings (when possible?) to capture comment lines in csv?

        # estimation.write_spec(simulate.read_model_spec(file_name=spec_file_name, tag='sample_spec')
        spec_file_name = model_settings[tag]

        # spec_df = simulate.read_model_spec(file_name=spec_file_name)
        # write_table(spec_df, tag, append=False)

        # if not spec_file_name.lower().endswith('.yaml'):
        #     file_name = '%s.yaml' % (spec_file_name, )

        input_path = config.config_file_path(spec_file_name)
        output_path = self.file_path(table_name=tag, file_type='csv')

        shutil.copy(input_path, output_path)


    def write_hook(self, df, table_name):
        if table_name == 'expression_values':
            # mergesort is the only stable sort, and we want the expressions to appear in original df column order
            index_name = df.index.name
            df = pd.melt(df.reset_index(), id_vars=[index_name]).sort_values(by=index_name, kind='mergesort')
            self.write_table(df, table_name, index=False)


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

            self.write_table(df, table_name, index=True)

        else:
            self.write_table(df, table_name)


    def get_override_choices(self, choices):
        assert self.estimating

        file_name = self.choice_settings['file_name']
        column_name = self.choice_settings['column_name']

        file_path = config.data_file_path(self.choice_settings['file_name'], mandatory=True)
        choice_df = pd.read_csv(file_path, index_col=0)

         # expect choice column <column_name> in choices table
        assert column_name in choice_df.columns, \
            "Column %s not in choice override table %s" % (column_name, file_name)

        assert isinstance(choices, pd.Series)
        override_choices = choices.to_frame('model_choice')
        override_choices['override_choice'] = reindex(choice_df[column_name], choices.index)

        self.write_table(override_choices, 'override_choices', index=True, append=True)
        return override_choices['override_choice']



manager = EstimationManager()

def write_hook(df, table_name):
    manager.write_hook(df, table_name)
