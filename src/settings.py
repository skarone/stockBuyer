
# INVESTMENT
# Fee for converting from pound to dollar.
CONVERSION_FEE = 0.005
# Fee for spread (Learn what is this)
SPREAD_FEE = 0.09
DOLLAR_TO_POUND = 0.7522
investment = 2000 #Pounds
pound_to_dollar = 1.0/DOLLAR_TO_POUND
investment_to_dollars = investment * 1.0/DOLLAR_TO_POUND
investment_after_conversion = investment*(pound_to_dollar-CONVERSION_FEE)
INVESTMENT = investment_after_conversion

PERIOD = "1d"
INTERVALS = "1m"