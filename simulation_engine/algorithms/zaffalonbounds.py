import jpype
import jpype.imports
from jpype.types import JArray, JByte
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import pandas as pd
from simulation_engine.util.alg_util import AlgUtil
import multiprocessing

class ZaffalonBounds:


    
    @staticmethod
    def bound_binaryIV(data, query, max_workers=None, isConf=False):

        # if max_workers is None, use the number of available CPUs
        if max_workers is None:
            max_workers = multiprocessing.cpu_count()


        row_dicts = [row.to_dict() for _, row in data.iterrows()]
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            func = partial(ZaffalonBounds._run_zaffalon_from_row_dict, query=query, isConf=isConf)
            results = list(executor.map(func, row_dicts))

        results_df = pd.DataFrame(results)
        data[results_df.columns] = results_df
        return data


    @staticmethod
    def _run_zaffalon_from_row_dict(row_dict, query, isConf=False):

        # try:
        if isConf:
            # For confounding variables, we only need X and Y
            df = pd.DataFrame({'Y': row_dict['Y'], 'X': row_dict['X']})
        else:
            df = pd.DataFrame({'Y': row_dict['Y'], 'X': row_dict['X'], 'Z': row_dict['Z']})
        bound_lower, bound_upper = ZaffalonBounds.run_experiment_binaryIV(query, df, isConf=isConf)

        failed = False

        # except Exception as e:
            # bound_lower = AlgUtil.get_trivial_Ceils(query)[0]
            # bound_upper = AlgUtil.get_trivial_Ceils(query)[1]
            # print(f"Error in Zaffalon: {e}")
            # failed = True

        #Flatten bounds to trivial ceils
        if failed:
            bound_upper = AlgUtil.get_trivial_Ceils(query)[1] 
            bound_lower = AlgUtil.get_trivial_Ceils(query)[0]

        bounds_valid = bound_lower <= row_dict[query+'_true'] <= bound_upper
        bounds_width = bound_upper - bound_lower

        return {
            query+'_zaffalonbounds_bound_lower': bound_lower,
            query+'_zaffalonbounds_bound_upper': bound_upper,
            query+'_zaffalonbounds_bound_valid': bounds_valid,
            query+'_zaffalonbounds_bound_width': bounds_width,
            query+'_zaffalonbounds_bound_failed': failed
        }



    @staticmethod
    def run_experiment_binaryIV(query, df, isConf=False):
        # Resolve path to this file
        this_dir = os.path.abspath(os.path.dirname(__file__))
        
        

        # Resolve jars relative to this file
        jar_zaffalon = os.path.join(this_dir, "zaffalon", "binaryIV", "zaffalon.jar")
        jar_credici = os.path.join(this_dir, "zaffalon", "credici.jar")
        if not jpype.isJVMStarted():
            jpype.startJVM(classpath=[jar_zaffalon, jar_credici])

        csv_data = ZaffalonBounds._dataframe_to_csv_string(df, isConf=isConf)


        ByteArrayInputStream = jpype.JClass("java.io.ByteArrayInputStream")
        input_bytes = JArray(JByte)(csv_data.encode('utf-8'))
        stream = ByteArrayInputStream(input_bytes)
        String = jpype.JClass("java.lang.String")
        query = String(query)

        BinaryTask = jpype.JClass("binaryIV.BinaryIVTask")
        task = BinaryTask(stream, query, jpype.JBoolean(isConf))
        result = task.call()
                
        # result looks like this: '-0.5813,-0.2671'
        # Convert to tuple of floats
        # print("Zaffalon result:", result)
        result_str = str(result)  # Convert java.lang.String to Python str

        if 'ERROR' in result_str:
            raise RuntimeError(f"Zaffalon Java returned an error: {result_str}")

        lower, upper = map(float, result_str.strip().split(","))
        return (lower, upper)


    @staticmethod
    def _dataframe_to_csv_string(df, isConf=False):
        if isConf:
            csv_data = "X,Y\n"
            for x, y in zip(df['X'].values, df['Y'].values):
                csv_data += f"{x},{y}\n"
            #write to CSV for testing
            # with open("test.csv", "w") as f:
            #     f.write(csv_data)
            return csv_data.strip()
            
        csv_data = "Z,X,Y\n"
        for z, x, y in zip(df['Z'].values, df['X'].values, df['Y'].values):
            csv_data += f"{z},{x},{y}\n"
        #write to CSV for testing
        # with open("test.csv", "w") as f:
        #     f.write(csv_data)
        return csv_data.strip()
