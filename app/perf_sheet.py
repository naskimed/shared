import pandas as pd


def get_perf(prices_df, starting_holdings, prices_bmk, simulationDates,portfolio_manager_behaviour):
	
    number_of_days = portfolio_manager_behaviour['style']['buy']['number_of_days']
    useful_prices = prices_df.loc[simulationDates[0]].T.to_frame(name='prices').dropna()
    holding = starting_holdings
    prices = prices_df
    pricesBMK = prices_bmk
	
    cols = set(useful_prices.index.values) - set(holding.index.values)
    cols = list(cols)

    #Instead of prices for one date(prev) we will extract all the simulation dates
    #We used to extract the data in the "current_date" and divide it by the data before "number_of_days" 
    #Now we will extract all the data after "number_of_days" ==> the data that we will devide it by the data since day one 
    #It's look like the inverse xD
    all_prices_after_number_of_days = prices.loc[simulationDates, cols].dropna().iloc[number_of_days:]
    all_prices = prices.loc[simulationDates, cols].dropna().iloc[:-number_of_days]
    first_div = all_prices_after_number_of_days.values / all_prices.values
    first_div = pd.DataFrame(first_div, columns=cols, index=simulationDates[number_of_days:])
            
            
    all_pricesBMK_after_number_of_days = pricesBMK.loc[simulationDates].dropna().iloc[number_of_days:]
    all_pricesBMK = pricesBMK.loc[simulationDates].dropna().iloc[:-number_of_days]
    second_div = all_pricesBMK_after_number_of_days.values / all_pricesBMK.values
            
    diff = first_div - second_div
    
    return diff