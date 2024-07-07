import time
import numpy as np
import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

starttime = time.time()

@dataclass
class Automate:
    yourDirName: str
    suffix: str
    cleanedcsvpath: str
    outlierPath: str

    def csvLocator(self):
        csvfiles = []
        for evfi in os.listdir(os.getcwd()):
            if evfi.endswith(self.suffix):
                csvfiles.append(evfi)
        return csvfiles

    def Relocater(self):
        """This method moves the CSV files into another special directory."""
        csv_files = self.csvLocator()
        if not os.path.exists(self.yourDirName):
            os.mkdir(self.yourDirName)
        for file in csv_files:
            os.rename(os.path.join(os.getcwd(), file), os.path.join(os.getcwd(), self.yourDirName, file))

    def nameCleaner(self):
        """This method cleans the name of the file and reads the values into a dictionary of dataframes."""
        df = {}
        for file in os.listdir(os.path.join(os.getcwd(), self.yourDirName)):
            cleaned_file_name = file.replace("-", "_").replace("/", "").replace("&", "").replace("$", "").replace("*", "_").replace(" ", "_")
            df[cleaned_file_name] = pd.read_csv(os.path.join(os.getcwd(), self.yourDirName, file), encoding="ISO-8859-1")
        return df

    def ColumnCleaner(self):
        """This method cleans column names in the dataframes."""
        cleaned_files = self.nameCleaner()
        for _, df_obj in cleaned_files.items():
            df_obj.columns = [col.replace("-", "_").replace("/", "").replace("&", "").replace("$", "").replace("*", "_").replace(" ", "_") for col in df_obj.columns]
        return cleaned_files

    def Outliers(self):
        """Use interquartile range for anomaly detection."""
        datafrms_dict = self.ColumnCleaner()
        cleaned_dict = {}
        outlier_dict = {}

        for df_name, dataframe in datafrms_dict.items():
            df = dataframe.copy()
            df_out = pd.DataFrame()

            for col in df.select_dtypes(include=[np.number]).columns:
                
                if df[col].dropna().empty:
                    continue

                q1, q3 = np.percentile(df[col].dropna(), [25, 75])
                iqr = q3 - q1
                lower_range = q1 - 1.5 * iqr
                upper_range = q3 + 1.5 * iqr

                outliers = df[(df[col] < lower_range) | (df[col] > upper_range)]
                df_out = pd.concat([df_out, outliers])

                df = df[(df[col] >= lower_range) & (df[col] <= upper_range)]
                df = df.drop_duplicates()

            cleaned_dict[df_name] = df
            outlier_dict[df_name] = df_out

        return cleaned_dict, outlier_dict

    def CleanedCsv(self):
        """This method saves cleaned data and outliers into CSV files."""
        if not os.path.exists(self.cleanedcsvpath):
            os.mkdir(self.cleanedcsvpath)
        
        cleaned_dict, outlier_dict = self.Outliers()
        for df_name, df in cleaned_dict.items():
            df.to_csv(os.path.join(os.getcwd(), self.cleanedcsvpath, df_name), header=True, index=False)

        if not os.path.exists(self.outlierPath):
            os.mkdir(self.outlierPath)
        
        for df_name, df in outlier_dict.items():
            df.to_csv(os.path.join(os.getcwd(), self.outlierPath, df_name), header=True, index=False)

    def splits(self, test_size, random_state, train_size):
        """This method splits the data into training and testing sets."""
        cleaned_dict, _ = self.Outliers()
        for df_name, df in cleaned_dict.items():
            cleaned_name = df_name.split(".")[0]
            file_name = f"{cleaned_name}'s_train_and_test_data"

            if not os.path.exists(os.path.join(os.getcwd(), self.cleanedcsvpath, file_name)):
                os.mkdir(os.path.join(os.getcwd(), self.cleanedcsvpath, file_name))

            train, test = train_test_split(df, test_size=test_size, train_size=train_size, random_state=random_state)
            train.to_csv(os.path.join(os.getcwd(), self.cleanedcsvpath, file_name, "train.csv"), header=True, index=False)
            test.to_csv(os.path.join(os.getcwd(), self.cleanedcsvpath, file_name, "test.csv"), header=True, index=False)

# Create an instance of the Automate class
automate = Automate('neededfiles', '.csv', 'cleaned', 'dirty')
automate.csvLocator()
automate.Relocater()
automate.nameCleaner()
cleaned_files = automate.ColumnCleaner()
cleaned_dict, outlier_dict = automate.Outliers()
automate.CleanedCsv()
automate.splits(test_size=0.3, random_state=40, train_size=0.7)

endtime = time.time()
print(f"The total execution time is {endtime - starttime} seconds")
