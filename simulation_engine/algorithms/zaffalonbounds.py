import jpype
import jpype.imports
from jpype.types import JArray, JByte
import os

class ZaffalonBounds:

    @staticmethod
    def run_experiment_binaryIV_ATE(df):
        # Resolve path to this file
        this_dir = os.path.abspath(os.path.dirname(__file__))

        # Resolve jars relative to this file
        jar_zaffalon = os.path.join(this_dir, "zaffalon", "binaryIV", "zaffalon.jar")
        jar_credici = os.path.join(this_dir, "zaffalon", "credici.jar")
        if not jpype.isJVMStarted():
            print("JVM started with classpath:", [jar_zaffalon, jar_credici])
            jpype.startJVM(classpath=[jar_zaffalon, jar_credici])

        csv_data = ZaffalonBounds._dataframe_to_csv_string(df)

        ByteArrayInputStream = jpype.JClass("java.io.ByteArrayInputStream")
        input_bytes = JArray(JByte)(csv_data.encode('utf-8'))
        stream = ByteArrayInputStream(input_bytes)

        BinaryTask = jpype.JClass("binaryIV.BinaryIVAteSimulationTask")
        task = BinaryTask(stream)
        result = task.call()
        # result looks like this: '-0.5813,-0.2671'
        # Convert to tuple of floats
        try:
            result_str = str(result)  # Convert java.lang.String to Python str
            lower, upper = map(float, result_str.strip().split(","))
            return (lower, upper)
        except Exception as e:
            raise ValueError(f"Failed to parse result '{result}': {e}")
        return (lower, upper)

    @staticmethod
    def _dataframe_to_csv_string(df):
        csv_data = "Z,X,Y\n"
        for z, x, y in zip(df['Z'].values, df['X'].values, df['Y'].values):
            csv_data += f"{z},{x},{y}\n"
        return csv_data.strip()
