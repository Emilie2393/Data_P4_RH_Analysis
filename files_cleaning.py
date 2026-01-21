import pandas as pd

class FilesCleaning():

    def __init__(self):
        self.extrait_eval_df = pd.read_csv("./raw_data/extrait_eval.csv")
        self.extrait_sirh_df = pd.read_csv("./raw_data/extrait_sirh.csv")
        self.extrait_sondage_df = pd.read_csv("./raw_data/extrait_sondage.csv")
        

    def doc_analysis(self):

        # On regarde comment un batiment est défini dans ce jeu de données 
        print("------\n", self.extrait_eval_df.head())
        print("------\n", self.extrait_sirh_df.head())
        print("------\n", self.extrait_sondage_df.head())

        # On regarde le nombre de valeurs manquantes par colonne ainsi que leur type 
        print("------\n", self.extrait_eval_df.info())
        print("------\n", self.extrait_sirh_df.info())
        print("------\n", self.extrait_sondage_df.info())

run = FilesCleaning()
run.doc_analysis()
        