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


class Estimator(object):

    def __init__(self, model_name, model_settings):

        logger.info("Initialize Estimator for'%s'" % (model_name,))

        self.model_name = model_name
        self.model_settings = model_settings
        self.estimating = True

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

        # FIXME - not required?
        # assert 'override_choices' in self.model_settings, \
        #     "override_choices not found for %s in %s." % (model_name, ESTIMATION_SETTINGS_FILE_NAME)

        self.omnibus_tables = self.model_settings.get('omnibus_tables', {})
        self.omnibus_tables_append_columns = self.model_settings.get('omnibus_tables_append_columns', [])
        self.tables = {}
        self.tables_to_cache = [table_name for tables in self.omnibus_tables.values() for table_name in tables]
        self.alt_id_column_name = None

    def log(self, msg, level=logging.INFO):
        logger.log(level, "%s: %s" % (self.model_name, msg))

    def set_alt_id(self, alt_id):
        self.alt_id_column_name = alt_id

    def get_alt_id(self):
        assert self.alt_id_column_name is not None, \
            "alt_id_column_name is None for %s did you forget to call set_alt_id()?" % (self.model_name, )
        return self.alt_id_column_name

    def end_estimation(self):

        self.write_omnibus_table()

        self.estimating = False
        self.tables = None

        logger.info("end estimation '%s'" % (self.model_name,))

        manager.release(self)

    def data_directory(self):

        # shouldn't be asking for this if not estimating
        assert self.estimating
        data_bundle_dir = config.output_file_path('estimation_data_bundle')

        return os.path.join(data_bundle_dir, self.model_name)

    def file_path(self, table_name, file_type=None):

        # shouldn't be asking for this if not estimating
        assert self.estimating

        if file_type:
            file_name = "%s_%s.%s" % (self.model_name, table_name, file_type)
        else:
            file_name = "%s_%s" % (self.model_name, table_name)

        return os.path.join(self.data_directory(), file_name)

    def write_table(self, df, table_name, index=True, append=True):

        def cache_table(df, table_name, append):
            if table_name in self.tables and not append:
                raise RuntimeError("cache_table %s append=False and table exists" % (table_name,))
            if table_name in self.tables:
                self.tables[table_name] = pd.concat([self.tables[table_name], df])
            else:
                self.tables[table_name] = df.copy()

        def write_table(df, table_name, index, append):
            file_path = self.file_path(table_name, 'csv')
            file_exists = os.path.isfile(file_path)
            if file_exists and not append:
                raise RuntimeError("write_table %s append=False and file exists: %s" % (table_name, file_path))
            df.to_csv(file_path, mode='a', index=index, header=(not file_exists))

        assert self.estimating

        if table_name in self.tables_to_cache:
            cache_table(df, table_name, append)
            logger.debug('write_table %s cache: %s' % (self.model_name, table_name))
        else:
            write_table(df, table_name, index, append)
            logger.debug('write_table %s write: %s' % (self.model_name, table_name))

    def write_omnibus_table(self):

        if len(self.omnibus_tables) == 0:
            return

        for omnibus_table, table_names in self.omnibus_tables.items():

            # ignore any ables not in cache
            table_names = [t for t in table_names if t in self.tables]
            concat_axis = 1 if omnibus_table in self.omnibus_tables_append_columns else 0

            df = pd.concat([self.tables[t] for t in table_names], axis=concat_axis)

            file_path = self.file_path(omnibus_table, 'csv')

            assert not os.path.isfile(file_path)

            df.sort_index(ascending=True, inplace=True, kind='mergesort')
            df.to_csv(file_path, mode='a', index=True, header=True)

            logger.debug('write_omnibus_choosers %s write: %s' % (self.model_name, file_path))

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
        self.write_table(coefficients_df, tag, append=False)

    def write_coefficients_template(self, coefficients_df, tag='coefficients_template'):
        assert self.estimating
        self.write_table(coefficients_df, tag, append=False)

    def write_choosers(self, choosers_df):
        self.write_table(choosers_df, 'choosers', append=True)

    def write_alternatives(self, alternatives_df):
        alternatives_df = self.melt_alternatives(alternatives_df)
        self.write_table(alternatives_df, 'alternatives', append=True)

    def write_choices(self, choices):

        # no need to write choices if also writing override_choices
        # if 'override_choices' in self.model_settings.get(self.estimating):
        #     return

        if isinstance(choices, pd.Series):
            choices = choices.to_frame(name='model_choice')
        self.write_table(choices, 'choices', append=True)

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

    def write_spec(self, model_settings=None, file_name=None, tag='SPEC'):

        if model_settings is not None:
            assert file_name is None
            file_name = model_settings[tag]

        input_path = config.config_file_path(file_name)
        output_path = self.file_path(table_name=tag, file_type='csv')

        shutil.copy(input_path, output_path)

        logger.debug("estimate.write_spec: %s" % output_path)

    def melt_alternatives(self, df):

        alt_id_name = self.alt_id_column_name

        assert alt_id_name is not None, \
            "alt_id not set. Did you forget to call set_alt_id()? (%s)" % self.model_name

        assert alt_id_name in df, \
            "alt_id_column_name '%s' not in alternatives table (%s)" % (alt_id_name, self.model_name)

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

    def write_interaction_expression_values(self, df):
        df = self.melt_alternatives(df)
        self.write_table(df, 'interaction_expression_values', append=True)

    def write_expression_values(self, df):
        self.write_table(df, 'write_expression_values', append=True)

    def get_override_choices(self, choices):
        """
        if choices is a series, then we label the model_choice and override_model_choice

        """
        assert self.estimating

        choice_settings = self.model_settings.get('override_choices')

        if choice_settings is None:
            logger.warning("estimation.get_override_choices - no override because no choice settings found for %s. ",
                           self.model_name)
            return choices

        # read override_df table
        file_name = choice_settings['file_name']
        file_path = config.data_file_path(file_name, mandatory=True)
        survey_df = pd.read_csv(file_path, index_col=0)

        if isinstance(choices, pd.Series):
            column_name = choice_settings['column_name']
            assert isinstance(column_name, str)

            override_choices = choices.to_frame('model_choice')
            override_choices['override_choice'] = reindex(survey_df[column_name], override_choices.index)

            # shouldn't be any choices we can't override
            if override_choices.override_choice.isna().any():
                missing_override_choices = override_choices[override_choices.override_choice.isna()]
                print("couldn't override choices for %s of %s choices\n" %
                      (len(missing_override_choices), len(override_choices)))
                print(missing_override_choices)
                bug

            self.write_table(override_choices, 'override_choices', append=True)
            return override_choices['override_choice']

        else:
            raise RuntimeError("get_override_choices does not grok choice data type")


class EstimationManager(object):

    def __init__(self):

        self.settings_initialized = False
        self.models = []
        self.model_settings = {}
        self.estimating = {}

    def initialize_settings(self):
        assert not self.settings_initialized
        settings = config.read_model_settings(ESTIMATION_SETTINGS_FILE_NAME)
        self.enabled = settings.get('enable', 'True')
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

        # global estimation setting
        if not self.enabled:
            return None

        if model_name not in self.models:
            return None

        # can't estimate the same model simultaneously
        assert model_name in self.model_settings, \
            "Cant begin estimating %s - already estimating that model." % (model_name, )

        assert model_name in self.model_settings, \
            "No estimation settings for %s in %s." % (model_name, ESTIMATION_SETTINGS_FILE_NAME)

        self.estimating[model_name] = Estimator(model_name, self.model_settings[model_name])

        return self.estimating[model_name]

    def release(self, estimator):

        self.estimating.pop(estimator.model_name)


manager = EstimationManager()
