import vaex
import numpy as np
import pandas as pd
#from slugify import slugify

@vaex.register_dataframe_accessor('cust1', override=True)
class cust1:
    def __init__(self, df):
        self.df = df

    def shift_constant(self, org_column, new_column, n, inplace=False, k=None):
        # NOT working...
        # make a copy without column
        df = self.df.copy()

        # make a copy with just the colum
        df_column = self.df[[org_column]]
        #print(df)
        # slice off the head and tail
        new_column_clean = new_column
        if n>0:
            df_head = vaex.from_dict({new_column_clean:[k]*n})
            df_tail = df_column[:-n]
            if org_column in df_tail.virtual_columns:
                # special handling because virtual columns carry around dependencies which make the concat fail
                df_tail = df_tail.materialize(org_column)
                df_tail = df_tail[[org_column]]
            df_tail.rename(org_column, new_column)
        else:
            df_tail = vaex.from_dict({ new_column_clean:[k]*abs(n)})
            df_head = df_column[-n:]
            if org_column in df_head.virtual_columns:
                df_head = df_head.materialize(org_column)
                df_head = df_head[[org_column]]
            df_head.rename(org_column, new_column)

        # stitch them together
        #print(df_head)
        #print(df_tail)
        df_shifted = df_head.concat(df_tail)
        #print(df_head)
        #print(df_tail)
        #print(df_shifted)
        # and join (based on row number)
        #df = df.join(df_shifted, inplace=False)
        self.df = self.df.join(df_shifted, inplace=True)
        return self.df #df_shifted[0]

    def shift_rotate(self, column, n, inplace=False):
        # make a copy without column
        df = self.df.copy().drop(column)
        # make a copy with just the colum
        df_column = self.df[[column]]
        # slice off the head and tail
        df_head = df_column[-n:]
        df_tail = df_column[:-n]
        # stitch them together
        df_shifted = df_head.concat(df_tail)
        # and join (based on row number)
        return df.join(df_shifted, inplace=inplace)
