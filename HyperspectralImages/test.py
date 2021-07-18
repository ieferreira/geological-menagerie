def check_unique_elements(ls1, ls2):
    """return elements contained in both lists
    """
    return list(set(ls1) & set(ls2))


def filter_dataframe(df, ls):
    """Keep rows of dataframe on which column "Linea" is in list ls
    """
    return df[df['Linea'].isin(ls)]


def set_dataframe(df):
    """Collapse dataframe by making a set of elements in column "Linea" and keeping first row
    """
    return df.groupby('Linea').first().reset_index()
    
