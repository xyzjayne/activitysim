# ActivitySim
# See full license in LICENSE.txt.

import os
import shutil

import logging

import yaml

import pandas as pd

from activitysim.core import config
from activitysim.core.util import reindex

logger = logging.getLogger('estimation')

ESTIMATION_SETTINGS_FILE_NAME = 'estimation.yaml'

class EstimationManager(object):

    def __init__(self):

        self.settings_initialized = False
        self.models = []
        self.model_settings = {}

        # name of model we are currently estimating, or None if not estimating
        self.estimating = None
        self.tables = None

    def initialize_settings(self):
        assert not self.settings_initialized
        settings = config.read_model_settings(ESTIMATION_SETTINGS_FILE_NAME)
        self.models = settings.get('models', [])
        self.model_settings = settings.get('model_settings', {})
        self.settings_initialized = True

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
        if not self.settings_initialized:
            self.initialize_settings()

        # shouldn't already be estimating
        assert self.estimating is None, \
            "Cant begin estimating %s - already estimating %s" % (model_name, self.estimating)

        if not model_name in self.models:
            return False

        # begin estimating model specified by model_name
        self.estimating = model_name
        logger.info("begin estimation '%s'" % (model_name,))

        # ensure there are choice override settings for this model
        if model_name not in self.model_settings:
            logger.warning("get_override_choices - no choice settings for %s in %s",
                         self.estimating, ESTIMATION_SETTINGS_FILE_NAME)

        # assert model_name in self.model_settings, \
        #     "Choice override settings for model %s not found in estimation settings file."

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

        self.tables = {}
        self.omnibus_tables = self.model_settings[model_name].get('omnibus_tables', [])
        self.omnibus_tables_concat_axis = 1 if self.model_settings[model_name].get('omnibus_tables_append_columns', 0) else 0

        return True

    def end_estimation(self):

        self.write_omnibus_table()

        logger.info("end estimation '%s'" % (self.estimating,))
        self.estimating = None
        self.tables = None

    def data_directory(self):

        # shouldn't be asking for this if not estimating
        assert self.estimating
        data_bundle_dir = config.output_file_path('estimation_data_bundle')

        return os.path.join(data_bundle_dir, self.estimating)

    def file_path(self, table_name, file_type=None):

        # shouldn't be asking for this if not estimating
        assert self.estimating

        if file_type:
            file_name = "%s_%s.%s" % (self.estimating, table_name, file_type)
        else:
            file_name = "%s_%s" % (self.estimating, table_name)

        return os.path.join(self.data_directory(), file_name)

    def cache_table(self, df, table_name, append=True):

        assert self.estimating

        if table_name in self.tables and not append:
            raise RuntimeError("cache_table %s append=False and table exists" % (table_name,))

        if table_name in self.tables:
            self.tables[table_name] = pd.concat([self.tables[table_name], df])
        else:
            self.tables[table_name] = df.copy()

        #print("cache_table", table_name, self.tables[table_name])

    def write_table(self, df, table_name, index=True, append=True):

        file_path = self.file_path(table_name, 'csv')

        file_exists = os.path.isfile(file_path)

        if file_exists and not append:
            raise RuntimeError("write_table %s append=False and file exists: %s" % (table_name, file_path))

        df.to_csv(file_path, mode='a', index=index, header=(not file_exists))

        logger.debug('estimate.write_table: %s' % file_path)

    def write_omnibus_table(self):

        if len(self.omnibus_tables) == 0:
            return

        omnibus_tables = [c for c in self.omnibus_tables if c in self.tables]

        df = pd.concat([self.tables[t] for t in omnibus_tables], axis=self.omnibus_tables_concat_axis)

        file_path = self.file_path('values_combined', 'csv')

        assert not os.path.isfile(file_path)

        df.sort_index(ascending=True, inplace=True, kind='mergesort')
        df.to_csv(file_path, mode='a', index=True, header=True)

        logger.debug('estimate.write_omnibus_choosers: %s' % file_path)


    def write_dict(self, d, dict_name):

        assert self.estimating

        file_path = self.file_path(dict_name, 'yaml')

        # we don't know how to concat, and afraid to overwrite
        assert not os.path.isfile(file_path)

        with open(file_path, 'w') as f:
            # write ordered dict as array
            yaml.dump(d, f)

        logger.debug("estimate.write_dict: %s" % file_path)


    def write_coefficients(self, coefficients_df, tag='coefficients'):
        assert self.estimating
        self.write_table(coefficients_df, tag, index=True, append=False)

    def write_choosers(self, choosers_df):
        self.cache_table(choosers_df, 'choosers', append=True)
        self.write_table(choosers_df, 'choosers', index=True, append=True)


    def write_alternatives(self, alternatives_df):
        #self.write_table(alternatives_df, 'alternatives', index=True, append=True)
        self.cache_table(self.melt_alternatives(alternatives_df, column_name='xxx'), 'alternatives', append=True)


    def write_choices(self, choices):
        if isinstance(choices, pd.Series):
            choices = choices.to_frame(name='choices')

        self.cache_table(choices, 'choices', append=True)


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

        logger.debug("estimate.write_spec: %s" % output_path)


    def melt_alternatives(self, df, column_name):

        variable_column = 'variable'

        #            alt_dest  util_dist_0_1  util_dist_1_2  ...
        # person_id                                          ...
        # 31153             1            1.0           0.75  ...
        # 31153             2            1.0           0.46  ...
        # 31153             3            1.0           0.28  ...
        # 31153             4            1.0           0.64  ...
        # 31153             5            1.0           0.48  ...

        # mergesort is the only stable sort, and we want the expressions to appear in original df column order
        index_name = df.index.name
        alt_id_name = 'alt_dest'
        melt_df = pd.melt(df.reset_index(), id_vars=[index_name, alt_id_name]) \
            .sort_values(by=index_name, kind='mergesort') \
            .rename(columns={'variable': variable_column})

        # person_id,alt_dest,expression,value
        # 31153,1,util_dist_0_1,1.0
        # 31153,2,util_dist_0_1,1.0
        # 31153,3,util_dist_0_1,1.0
        # 31153,4,util_dist_0_1,1.0

        melt_df = melt_df.set_index([index_name, variable_column, alt_id_name]).unstack(2)
        melt_df.columns = melt_df.columns.droplevel(0)
        melt_df = melt_df.reset_index(1)

        # person_id,expression,1,2,3,4,5,...
        # 31153,util_dist_0_1,0.75,0.46,0.27,0.63,0.48,...
        # 31153,util_dist_1_2,0.0,0.0,0.0,0.0,0.0,...
        # 31153,util_dist_2_3,0.0,0.0,0.0,0.0,0.0,...

        return melt_df

    def write_hook(self, df, table_name):

        if table_name == 'expression_values':

            self.cache_table(df, table_name, append=True)


        elif table_name == 'interaction_expression_values':

            df = self.melt_alternatives(df, column_name='utility_expression')
            #self.write_table(df, table_name, index=True, append=True)
            self.cache_table(df, table_name, append=True)

        else:
            self.write_table(df, table_name)


    def get_override_choices(self, choices):
        """
        if choices is a series, then we label the model_choice and override_model_choice

        """
        assert self.estimating

        choice_settings = self.model_settings.get(self.estimating)

        if choice_settings is None:
            logger.warn("estimation.get_override_choices - no override because no choice settings found for %s. ",
                         self.estimating)
            return choices

        # read override_df table
        file_name = choice_settings['file_name']
        file_path = config.data_file_path(file_name, mandatory=True)
        survey_df = pd.read_csv(file_path, index_col=0)

        column_names = choice_settings['column_name']

        if isinstance(choices, pd.Series):
            column_name = choice_settings['column_name']
            assert isinstance(column_name, str)

            override_choices = choices.to_frame('model_choice')
            override_choices['override_choice'] = reindex(survey_df[column_names], override_choices.index)

            self.cache_table(override_choices, 'override_choices', append=True)
            return override_choices['override_choice']

        elif isinstance(choices, pd.Series):
            assert isinstance(choices, pd.DataFrame)

            # FIXME - column_names list not really needed, but nice documentation of requirements
            # should be same number of overrides as choice columns
            assert set(column_names) == set(choices.columns),\
                "column_name list (%s) does not match choices columns (%s)" % (column_names, choices.columns)

            # expect choice column <column_name> in choices table
            assert set(column_names).issubset(set(survey_df.columns)),\
                "Missing choices columns (%s) in verride table %s" % \
                (set(column_names) - set(survey_df.columns), file_name)

            # copy in the overrides with desired column names
            override_choices = pd.DataFrame(index=choices.index)
            for c in column_names:
                override_choices[c] = reindex(survey_df[c], override_choices.index)

            # write table with both 'modeled_' and the 'override_' columns
            bundle_df = pd.concat(
                [choices.rename(columns={c: 'modeled_' + c for c in choices.columns}),
                override_choices.rename(columns={c: 'override_' + c for c in choices.columns})]
            )
            self.cache_table(df=bundle_df, table_name='override_choices', append=True)

            # print(choices)
            # print(override_choices)
            # bug

            return override_choices
        else:
            raise RuntimeError("get_override_choices does not grok choice data type")



manager = EstimationManager()

def write_hook(df, table_name):
    manager.write_hook(df, table_name)
