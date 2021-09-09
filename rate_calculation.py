
def _growth_rate( v1, v2 ):
    """
    Function that cal culated the percentage differnece 
    of 'V2' compared to 'V1'
    """
    result = (( v2 - v1 )/ v1) * 100 
    return result

def compute_yearly_growth_rates(
    df, 
    years_selected, 
    base_year
):
    """
    Takes data frame 'df' as input and 
    returns growth rates 'values1' and 'values2'
    only for the years in 'years_selected'

    """
              
    values1= []
    values2 = []
    for year in years_selected:
        # growth rate - compared to  the base year 
        values1.append(_growth_rate(df[df.index == base_year].iloc[0], df[df.index == year].iloc[0]))

        # growth rate - compared to the immediate previous year in 'years_selected'
        values2.append(_growth_rate(df[df.index == year-1].iloc[0], df[df.index == year].iloc[0]))

    return values1, values2


def format_data_yearwise(df):
    """
    Function that generates and returns yearwise monthly/weekly data 

    Returns data in below format
    --------------------------------
    Format:
    { 2019 : monthly_data(dataframe),
      2020 : monthly_data(dataframe) }
    OR
    { 2019 : weekly data(dataframe),
      2020 : weekly_data(dataframe) }
    """
    # get years
    years = df.index.year.unique()
    data ={}

    for year in years:
        # year wise data
        data[year] = df.loc[df.index.year == year]

    return data

def get_growth_rate(df1, df2): 
    """
    Calculates and returns growth rates of df2 compared to df1 
    
    """ 
    gr_month = []
    for i in range(len(df1.values)):
        try:
            gr = _growth_rate(df1.iloc[i], df2.iloc[i])
            gr_month.append(gr) 
        except:
            gr_month.append(0)
    return gr_month


def calculate_growth_rates_with_base_year(
    base_year, 
    years_to_compare, 
    data, 
    typ
):
    """
    Calculates and returns monthly/weekly growth rates for the years in 'years_to_compare'
    compared to base year in 'base_year'

    Parameters
    ----------
    base_year : int, year value, based on this 'base_year' value the 
    years_to_compare: list, years for which the monthly/weekly growth rate is calculated 
    data: dictonary, year wise values of energy - {2014 : energy_df1, 2015 : energy_df2}, 
    typ: data type Demand/Wind_G

    """ 
    
    gr = {}

    for k, v in data.items():
        
        # compute only for given years
        if k not in years_to_compare or k == base_year:
            continue
        
        # get growth rate for parameters based on 'typ'
        gr[k] = get_growth_rate( data[base_year][typ], v[typ] )

    return gr

def calculate_growth_rates_with_prev(
    year, 
    data, 
    typ,
    last_check =True
):
    """
    Function that calculates and returns the monthly/weekly growth rates for 'year'
    compared to the previous month/week of the same year

    Parameters
    ----------
    year: int, yer value for wich the monthly/weekly growth rate is clculated
    data: dictonary, year wise values of energy - {2014 : energy_df1, 2015 : energy_df2}, 
    typ: data type Demand/Wind_G
    last_check : if last week (52) or month(12) is passed then set to TRUE else 'FALSE'
    """ 
    gr = []

    ln = len(data[year][typ])

    if last_check == True :
        gr.append(_growth_rate(data[year - 1][typ].iloc[ln - 1], data[year][typ].iloc[0]) )
    # growth rate of 'year' - month/week wise 
    for i in range(ln  - 1):
        gr.append(_growth_rate(data[year][typ].iloc[i], data[year][typ].iloc[i+1]))
    
    return gr
