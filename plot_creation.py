import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


# Constants
PLTTYPE = "ggplot"
TXT_BOTTOM = "bottom"
TXT_TOP = "top"
TXT_GROWTH_RATE = 'Growth Rate(%)'
TXT_YEAR = 'Year'
TXT_MONTH = 'Month'

def _create_bar( 
    ax, 
    x_loc, 
    bar_values, 
    bar_label=None, 
    barwidth=0.2, 
    bar_color='blue', 
    ha='center', 
    rot=0
):

    """
        To Create bars in bar plot and position the values(bartext) 

        Parameters
        ----------
        'ax' : bar axes (individual subplots) ;
        'x_labels' : bar loction along x axis ;
        'bar_values' : list of values for barplot ;
        'bar_label' : sets label for bar ;
        'barwidth' : width of the bar ;
        'bar_color' : seta the bar color - default value 'blue' ;
        'ha' : horizontalalignment of bar text- default value 'center' ;
        'rot' : rotation of bar text - default value 0
    """

    ax.bar(x_loc, bar_values, color=bar_color, width=barwidth, label=bar_label)
    # put bar values ( upto 2 decimals) in the plot
    # top the bar- in case of possitive values
    # bottom of the bar - in case of negative values 
    for i, v in enumerate(bar_values):

        if v >= 0:
            p = v + 0.2
            va = TXT_BOTTOM
        else:
            p = v - 0.5
            va = TXT_TOP

        ax.text(x_loc[i] - 0.1, p, str(round(v, 2)), rotation=rot, horizontalalignment=ha, verticalalignment=va, fontsize=15) #weight='bold'

def _set_axes( 
    ax, 
    title=None, 
    xlabel=None, 
    ylabel=None, 
    xticks=None, 
    xticklabels=None, 
    legend_loc=None 
    ):

    """
        To set axes in plot  

        Parameters
        ----------
        'ax' : bar axes (individual subplots) ;
        'title' : Title of the plot/subplot ;
        'xlabel' : x axis label ;
        'ylabel' : y axis label ;
        'xticks' : ticks in x axis ;
        'xticklabels' : labels of xticks ;
        'legend_loc' : location of legend
    """
    ax.legend( loc=legend_loc, fontsize=14)
    ax.set_title(title, fontsize=16 )

    # set xlabel
    ax.set_xlabel(xlabel, fontsize=16, fontweight= 'bold' )
  
    #set ylabel
    ax.set_ylabel(ylabel, fontsize=16, fontweight= 'bold' )

    # x ticks
    if xticks :
        ax.set_xticks(xticks) 
    if xticklabels :
        ax.set_xticklabels(xticklabels, fontsize=14, fontweight= 'bold')
    else:
        ax.tick_params(axis='x', labelsize=14  )

     # y ticks   
    ax.tick_params(axis='y', labelsize=14  )

def plot_yearly_data(
    demand, 
    wind, 
    figsize=None, 
    label_demand=None, 
    label_wind=None, 
    title=None, 
    ylim=None
):
    """
        Uses to plot yearly Deamnd and Wind energy data.

        Parameters
        ----------
        'demand' : Energy demand(yearly) ; 
        'wind' : Generated Wind energy(yearly) ; 
        'figsize' : size of the plot ; 
        'label_demand' : label for supplied 'demand' - default None ;
        'label_wind' : label for supplied 'wind' - default None ;
        'title' : title of the plt ;
        'ylim' : limit along y axis 
    """
    plt.style.use(PLTTYPE)
    plt.figure(figsize=(figsize))
    plt.plot(demand, marker = '.', label=label_demand) 
    plt.plot(wind, marker = '.', label=label_wind) 

    plt.legend(fontsize=14)
    plt.ylim(ylim)
    plt.xlabel(TXT_YEAR, fontsize=14, fontweight='bold')
    plt.ylabel("Energy(MW)", fontsize=14, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xticks(fontsize=14 , fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    plt.show()
  

def plot_yearly_growth_rate( 
    bar_values1, 
    base_year, 
    bar_values2, 
    suptitle=None, 
    xticklabels=None, 
    barwidth=0.2, 
    ha='center', 
    rot=0, 
    ylim=None, 
    figsize=None, 
    legend_loc=None
):
    """
        To plot(barchart) yearly Growth rates  
        First subplot plots barchart of 'bar_values1' 
        Second subplot plots barchart of 'bar_values2'

        Parameters
        ----------
        'bar_values1' : List- Growth rate compared to 'base_year' ;
        'base_year' : int -base year for 'bar_values1' ;
        'bar_values2' : List - Growth rate compared to the previous year ;
        'suptitle' : Title of the Figure ;
        'xticklabels' : labels of xticks ;
        'barwidth' : width of the bars- default 0.2;
        'bar_color' : set the bar color ;
        'ha' : horizontalalignment of bar text- default value 'center' ;
        'rot' : rotation of bar text - default value 0 ;
        'ylim' : range accross y axis ;
        'figsize' : size of the plot ;
        'legend_loc' : Location of legend 

    """
    
    plt.style.use(PLTTYPE)
    # Create a containing figure with 1x2 arrangement of 'axes' (individual subplots).
    fig, ax = plt.subplots(1, 2,  sharey=True, figsize = figsize)
    fig.suptitle(suptitle, fontsize=18, fontweight='bold')

    x_loc= np.arange(len(xticklabels))

    # Growth rate compared to - 'base_year'
    _create_bar(   ax=ax[0], 
                    x_loc=x_loc, 
                    bar_values=bar_values1, 
                    bar_label='Growth rate - compared to year %s' %(base_year), 
                    barwidth=barwidth, 
                    bar_color='blue',
                    ha = ha, 
                    rot=rot )

    _set_axes(  ax=ax[0], 
                title='Growth rate - compared to year %s' %(base_year), 
                xlabel=TXT_YEAR, 
                ylabel=TXT_GROWTH_RATE, 
                xticks=x_loc, 
                xticklabels=xticklabels,
                legend_loc = legend_loc )

    # Growth rate compared to the previous year
    _create_bar(    ax=ax[1], 
                    x_loc=x_loc, 
                    bar_values=bar_values2, 
                    bar_label='Growth rate - compared to the previous year', 
                    barwidth=barwidth, 
                    bar_color='red',
                    ha = ha, 
                    rot=rot )

    _set_axes(  ax=ax[1], 
                title='Growth rate - compared to the previous year', 
                xlabel=TXT_YEAR, 
                ylabel=TXT_GROWTH_RATE, 
                xticks=x_loc, 
                xticklabels=xticklabels,
                legend_loc = legend_loc )
            
    plt.ylim(ylim)
    plt.yticks(fontsize=14 , fontweight='bold')
    # Allow more space around subplots.
    fig.tight_layout()
    plt.show()

def create_line_plot(
    df, 
    selected_years, 
    v_loc, 
    typ, 
    title=None, 
    xlabel=None, 
    ylabel=None, 
    xticks=None, 
    xticklabels=None, 
    figsize=None, 
    xlim=None, 
    ylim=None,
    xtick_rot = 0
):
    """
        Function to generate yearwise line plots - with different Lockdown stages in 'v_loc'
        
        Parameters
        ----------
        'df' : Dataframe -values to plot - (daily/weekly/monthly -data) ;
        'selected_years' : List of years - line plot will only be gererated for the years entered  ;
        'v_loc' : List - locations of the verticle lines in the plot -[indication of diff lockdown stages] ;
        'typ' : type of the data to be ploted (Demand/Wind)
        'title' : Title of the plot ;
        'xlabel' : x axis label ;
        'ylabel' : y axis label ;
        'xticks' : ticks in x axis ;
        'xticklabels' : labels of xticks ;
        'figsize' : size of the plot ;
        'xlim' : range accross x axis ;
        'ylim' : range accross y axis ;
        'xtick_rot' : xtick rotaion 

    """
    plt.style.use(PLTTYPE)
    plt.figure(figsize=figsize)

    for year in selected_years: # plot each year
        data = df[df.index.year == year]

        if year == 2020:
            plt.plot(np.arange(len(data))+1, data[typ],color='k', linestyle='-.', label=year) 
        else:
            plt.plot(np.arange(len(data))+1, data[typ], label=year) 

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(xticks, xticklabels, fontsize=14 , fontweight='bold', rotation=xtick_rot)
    plt.yticks(fontsize=14 , fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold' )
    plt.xlabel(xlabel, fontsize=14 , fontweight='bold')
    plt.ylabel(ylabel, fontsize=14 , fontweight='bold')


    # create color mapping for vertical lines
    v_colors = sns.color_palette('dark', n_colors=len(v_loc))  # get a number of colors for vertical lines
    v_cmap = dict(zip(v_loc.keys(), v_colors))  # zip values to colors for vertical lines

    # vertical lines- indication of lockdown stages
    for k, v in v_loc.items():
        plt.vlines(x=v, ymin = min(ylim), ymax = max(ylim), colors=v_cmap[k], linestyle='--', label=k)

    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, fontsize=14)
    plt.show()


def plot_growth_rates( 
    gr1, 
    years_to_compare, 
    base_year, 
    gr2, 
    gr2_y, 
    typ, 
    nrows=1, 
    ncols=2, 
    barwidth=0.2, 
    title=None, 
    ha='center', 
    rot=0, 
    xtick_labels=None, 
    figsize=None, 
    ylim=None, 
    legend_loc=None
):
    """
        To plot(barchart) monthly/weekly Growth rates  
        First subplot plots barchart of 'gr1' 
        Second subplot plots barchart of 'gr2'

        Parameters
        ----------
        'gr1' : Dictionary- Growth rates years in 'years_to_compare' compared to 'base_year' - 
                Format{ 2018 : list(montly_growthrate), 
                        2019 : list(montly_growthrate)
                        } ;
        'years_to_compare' : List - selected years, for which the growth rates are ploted ;
        'base_year' : int - base year - the yearly growth rates in barchart is compared to 'base_year' ;
        'gr2' : List - Growth rate of the year 'gr2_y' compared to the previous month/week ;
        'gr2_y' : int - year, for which the growth rates 'gr2' will be ploted ;
        'typ' : monthly data / weekly data values('Month', 'Week') ;
        'nrows' : int, default 1 - Number of rows/columns of the subplot grid ;
        'ncols' : int, default 2 - Number of columns of the subplot grid ;
                 ( only two possibilities her - if nrows eq 1 then ncols = 2 / if nrows eq 2 then  ncols eq 1)
        'barwidth' : width of the bars - default 0.2 ;
        'title' : Title of the Plot ;
        'ha' : horizontalalignment of bar text- default value 'center' ;
        'rot' : rotation of bar text - default value 0 ;
        'xtick_labels' : labels of xticks ;
        'figsize' : size of the plot ;
        'ylim' : range accross y axis ;
        'legend_loc' : Location of legend 
    
    """     
    colors = sns.color_palette('husl', n_colors=len(years_to_compare))  # get a number of colors
    cmap = dict(zip(years_to_compare, colors))  # zip values to colors

    plt.style.use(PLTTYPE)
    # Create a containing figure with 1x2 arrangement of 'axes' (individual subplots).
    fig, ax = plt.subplots(nrows, ncols,  sharey=True, figsize = figsize )
    fig.suptitle(title , fontsize=20 , fontweight='bold')

    bar = np.arange(len(xtick_labels))

    # bars for Growth rate compared to - 'base_year'
    bar1 = bar
    for year in years_to_compare:

        _create_bar(    ax=ax[0], 
                        x_loc=bar1 , 
                        bar_values=gr1[year], 
                        barwidth=barwidth, 
                        bar_label='%sly Growth rate of %s - compared to %s' %(typ, year, base_year), 
                        bar_color= cmap[year],
                        ha = ha, 
                        rot=rot )

        bar1 = [i+barwidth for i in bar1]
        

    _set_axes(  ax=ax[0],
                title='Growth rate -  compared to the same %s of the year %s' %(typ, base_year),
                xlabel=typ,
                ylabel=TXT_GROWTH_RATE,
                xticks=bar + barwidth*(len(years_to_compare)/3),
                xticklabels = xtick_labels ,
                legend_loc = legend_loc )

    # bars for Growth rate of year 'gr2_y' compared to the previous month
    _create_bar(    ax=ax[1], 
                    x_loc=bar, 
                    bar_values=gr2, 
                    barwidth=barwidth, 
                    bar_label='%sly Growth rate of %s - compared to the previous %s' %(typ, gr2_y, typ), 
                    bar_color='red',
                    # ha = ha, 
                    # rot=rot
                )

    _set_axes(  ax=ax[1],
                title='Growth rate %s - compared to the previous %s' %(gr2_y, typ) ,
                xlabel=typ,
                ylabel=TXT_GROWTH_RATE,
                xticks=bar,
                xticklabels = xtick_labels ,
                legend_loc = legend_loc )

    plt.ylim(ylim)
    # Allow more space around subplots.
    fig.tight_layout()
    plt.show()

# Create a function to plot time series data
def plot_time_series(timesteps, values, title=None, format='-', start=0, end=None, label=None, xlim=None):
    """
    Plots a timesteps (a series of points in time) against values (a series of values across timesteps).
  
    Parameters
    ---------
    timesteps : array of timesteps
    values : array of values across time
    title : title of the plot
    format : style of plot, default "-"
    start : where to start the plot (setting a value will index from start of timesteps & values)
    end : where to end the plot (setting a value will index from end of timesteps & values)
    label : label to show on plot of values
    xlim : limit around x axis
    """
    # Plot the series
    plt.style.use("ggplot")
    plt.plot(timesteps[start:end], values[start:end], format, label=label)
    plt.title(title, fontsize=16, fontweight='bold')
    if xlim:
        plt.xlim(xlim)
    plt.xlabel("Time", fontsize=14, fontweight='bold')
    plt.ylabel("Energy(MW)", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=14 , fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    if label:
        plt.legend(fontsize=14) # make label bigger
    plt.grid(True)


def plot_normality_check( 
    data, 
    figsize = (15, 4),
    title=None, 
    ylim=None, 
    plot_type='H' 
):
    """
    Uses to check normality assumption by ploting Histogram and Q-Q plot

    Parameters
    ----------
    data: data in key value format(dictonary)
    figsize:  size of the plot
    title: Title of the plot
    ylim : limit accross y axix
    plot_type: default 'H'( histogram) [ possible values 'H' and 'Q'(QQ-plot)]

    Plots histogram / QQ-plots
    """  
    # Create a containing figure with 1xn arrangement of 'axes' (individual subplots).
    fig, ax = plt.subplots(1, len(data), sharey=True, figsize = figsize)

    # Add title at the top.
    fig.suptitle(title, fontsize=16, fontweight='bold')

    i = 0
    for k, v in data.items():

        if plot_type == 'H':
            ax[i].hist(v, color='b', bins = 20)
        elif plot_type == 'Q':
            stats.probplot(v, dist="norm", plot=ax[i])
        else:
            print('ERROR: Not a valid plot_type - accepted values H or Q')

        ax[i].set_title(k, fontsize=14, fontweight='bold')
        i += 1

    plt.ylim(ylim)
    # Allow more space around subplots.
    fig.tight_layout()
    plt.show()

def plot_predictions(
    df,
    rnn_results,
    lstm_results,
    EIR_results = None,
    title = None,
    figsize=(15, 6),
    anot_cord = (0.5, 0.8)
):
    """
    Function to plot prediction of the models 

    Parameters
    ----------
    df: data fram containing different prediction and actual value
    figsize:  size of the plot
    title: Title of the plot
    ylim : limit accross y axix
    plot_type: default 'H'( histogram) [ possible values 'H' and 'Q'(QQ-plot)]

    Plots histogram / QQ-plots
    """ 
    plt.style.use("ggplot")
    df.plot(figsize=figsize)
    plt.title(title, fontsize=18)
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0,fontsize=16)

    textstr1 = "RNN Model MAE = " + "{:.3f}".format(rnn_results['mae']) + " RNN Model RMSE = " +  "{:.3f}".format(rnn_results['rmse'])
    textstr2 = "\nLSTM Model MAE = " + "{:.3f}".format(lstm_results['mae']) + " LSTM Model RMSE = " +  "{:.3f}".format(lstm_results['rmse'])
    if EIR_results :
        textstr3 = "\nEIR forecast MAE = " + "{:.3f}".format(EIR_results['mae']) + " EIR forecast RMSE = " +  "{:.3f}".format(EIR_results['rmse'])
    else :
        textstr3 = ''   

    plt.annotate(textstr1 + textstr2+textstr3 , xy=anot_cord, xycoords='figure fraction', fontsize=12, weight='bold')
    plt.ylabel('Energy(MW)', fontsize=16)
    plt.xlabel('Day of the month', fontsize=16)
    plt.xticks(fontsize=14 , fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
