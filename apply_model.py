from pyspark import SparkFiles, SparkContext
from pyspark.sql import SQLContext, Row
from sklearn.externals import joblib
from time import strftime, gmtime

import pandas as pd
import os
import logging
import sys # Put at top if not already there

logger = logging.getLogger('py4j')
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.DEBUG)
logger.addHandler(sh)


class ModelApplier:
    def __init__(self, model_location_dict, input_location, output_location):
        self.model_location_dict = model_location_dict
        self.input_location = input_location
        self.output_location = output_location

    def get_partition_processor(self):

        def build_dataframe(textline):
            """
            Convert text line to dataframe.
            """
            column_names = []
            records = [line.split(u',') for line in textline]
            records = [pd.np.nan if token in (u'\\N', 'NULL') else token for token in records]
            # df_line = pd.read_csv(textline, header=None, names=column_names)
            df = pd.DataFrame(records, columns=column_names)
            df = df.convert_objects(convert_numeric=True)
            df.set_index('msisdn', inplace=True)
            print('-----', df.dtypes)
            return df

        def partition_processor(partitionlinechunks):
            """
            Partition logic for pyspark parallel processing
            """

            model_pipe_object = joblib.load(SparkFiles.get("mmp_phase1_D2.clf"))

            def set_predictions(x):
                segment = model_pipe_object.predict_proba(x)
                return segment

            df_with_nan = build_dataframe(partitionlinechunks)
            df_with_newline = df_with_nan.replace(u"NULL", pd.np.nan)
            behaviour_df = df_with_newline.replace(u"\\N", pd.np.nan)
            predictions_ser = set_predictions(behaviour_df)

            predictions_list = [value for value in [zip(predictions_ser.index, predictions_ser.loc[:,'A'], predictions_ser.loc[:,'Y'], predictions_ser.loc[:,'segment'], predictions_ser.loc[:,'model_version'])]]
            return iter(predictions_list)

        return partition_processor

    @staticmethod
    def add_files_to_context(location, sc):
        # TODO: move absoluteFilePaths to util.py
        def absoluteFilePaths(directory):
            for dirpath, _, filenames in os.walk(directory):
                for f in filenames:
                    yield os.path.abspath(os.path.join(dirpath, f))

        files = absoluteFilePaths(location)
        map(lambda x: sc.addFile(x), files)

    def apply(self):
        """
        Set context files, map partitions
        """

        sc = SparkContext(appName="Model Applier")
        sqlContext = SQLContext(sc)

        # Add model and supporting files to SparkContext
        for item in self.model_location_dict.items():
            ModelApplier.add_files_to_context(item[1], sc)

        partition_processor = self.get_partition_processor()
        infile = sc.textFile(self.input_location)
        header_line = infile.first()
        infile = infile.filter(lambda x: x != header_line)

        result = infile.mapPartitions(partition_processor).flatMap(lambda x: x)
        print('result.class', result.__class__)

        result = result.map(lambda (x, a, y, segment, model_version):
                            (int(x), float(a), float(y), segment, model_version))
        sqlContext.createDataFrame(result).saveAsParquetFile(self.output_location)


if __name__ == '__main__':
    models = {'tagging': ''}    
    input_location = ""
    output_location = "

    ModelApplier(models, input_location, output_location).apply()
    print('Before Start:', strftime('%Y-%m-%d %H:%M:%S', gmtime()))
    print('After Finish:', strftime('%Y-%m-%d %H:%M:%S', gmtime()))

    # from pyspark.sql import SQLContext, Row
    # sqlContext = SQLContext(sc)
    # parquetFile.collect()
