"""
Created on Mon May  5 19:47:34 2025
"""

from entsoe import EntsoeRawClient
import pandas as pd
import xml.etree.ElementTree as ET
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

###::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# INITIALIZE CLIENT
###::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
# insert your personal API key
API_TOKEN = 'YOUR_API_KEY'
client = EntsoeRawClient(api_key=API_TOKEN)

###::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# INPUTS
###::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
# Desired Location/Bidding Zone
country_code = 'IT_CNOR' 
time_zone = 'Europe/Rome'

# Historical Data Window
end_year = 2025
month = 4 
num_years = 2

###::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# API CALL: Historical Data Retrival 
###::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Since ENTSOE can't handle large api calls more than a year in lenght it's
# better to call each month independently then compile them all into one dataframe
start_year = end_year - num_years
all_data = []

# Loop through each month between start_year and end_year
current = pd.Timestamp(f"{start_year}-{month}-01", tz=time_zone)  # Start from April 1, 2023
end = pd.Timestamp(f"{end_year}-{month}-01", tz=time_zone)        # End at April 1, 2025

while current < end:
    next_month = current + relativedelta(months=1)

    print(f"Fetching data from {current} to {next_month}...")

    try:
        raw_xml = client.query_day_ahead_prices(country_code, current, next_month)
        
        # Parse XML
        root = ET.parse(StringIO(raw_xml)).getroot()
        ns = {'ns': root.tag.split('}')[0].strip('{')}  # extract namespace

        # Navigate to timeseries points
        for timeseries in root.findall('.//ns:TimeSeries', ns):
            for period in timeseries.findall('ns:Period', ns):
                start_time = pd.Timestamp(period.find('ns:timeInterval/ns:start', ns).text)
                resolution = period.find('ns:resolution', ns).text
                step = pd.Timedelta(resolution.replace('PT', '').lower())

                for point in period.findall('ns:Point', ns):
                    position = int(point.find('ns:position', ns).text)
                    price = float(point.find('ns:price.amount', ns).text)

                    timestamp = start_time + (position - 1) * step
                    all_data.append((timestamp, price))

    except Exception as e:
        print(f"Failed for {current.strftime('%Y-%m')}: {e}")

    current = next_month  # Move to the next month

# Error in case there is no data for the time period requested
if not all_data:
    raise RuntimeError(
        "No data returned for the requested window. "
        "Check API token validity, country_code/bidding zone coverage, "
        "and that ENTSO-E server is running."
    )

# Convert to DataFrame
df = pd.DataFrame(all_data, columns=['datetime', 'price_EUR_per_MWh'])
df.sort_values('datetime', inplace=True)
df.reset_index(drop=True, inplace=True)

# Optional: Save to CSV
# df.to_csv('insert_your_own_fp.csv', index=False)

print("Done. Preview:")
print(df.head())

###::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# CALCULATE VOLATILITY
###::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Assuming df has already been loaded and 'datetime' is parsed

# Drop null or zero columns
zero_prices = df[df['price_EUR_per_MWh'] == 0]
print(zero_prices) # This is just so you can do a quick gut check
df = df[df['price_EUR_per_MWh'] > 0]  # Remove zero or negative prices

# Calculate log returns
df['log_return'] = np.log(df['price_EUR_per_MWh'] / df['price_EUR_per_MWh'].shift(1))

# Drop NaN
df.dropna(inplace=True)

# Hourly volatility (standard deviation of returns)
hourly_vol = df['log_return'].std()

annualize_factor = df.shape[0]/num_years
annualized_vol = hourly_vol * np.sqrt(annualize_factor)

print(f"Hourly volatility: {hourly_vol:.6f}")
print(f"Annualized volatility: {annualized_vol:.6f}")

###::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# VISUALIZE: Plot Historical Hourly Spot Prices & Rolling Volatility
###::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
# Plot spot prices over time
plt.figure(figsize=(14, 5))
plt.plot(df['datetime'], df['price_EUR_per_MWh'], label='Spot Price (EUR/MWh)', color='#0b2667')
plt.title(f'{country_code} Day-Ahead Spot Prices (EUR/MWh)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Axis Limits
plt.xlim(left=df['datetime'].min(), right=df['datetime'].max())

# Style: remove grid and spines except bottom/left
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()

# Calculate rolling volatility (7-day window, assuming hourly data)
df['rolling_volatility'] = df['log_return'].rolling(window=24 * 7).std()

# Plot rolling volatility
plt.figure(figsize=(14, 5))
plt.plot(df['datetime'], df['rolling_volatility'], label='7-Day Rolling Volatility', color='red')
plt.title(f'{country_code} Rolling Volatility of Spot Prices (7-Day Window)')
plt.xlabel('Date')
plt.ylabel('Log Return Volatility')
plt.grid(True)
plt.legend()
plt.tight_layout()

# Optional: limit x-axis (commented out if unnecessary)
# plt.xlim(left=df['datetime'].min(), right=df['datetime'].max())

# Style: remove grid and spines except bottom/left
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

###::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# MONTE CARLO X GBM SIMULATION FOR FORECASTING
###::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Monte Carlo forecast parameters
num_simulations = 1000 
forecast_horizon_days = 14  # maximum days GBM model should be used to forecast
hours_per_day = 24
forecast_horizon_hours = forecast_horizon_days * hours_per_day
np.random.seed(42)  # Simulation seed for reproducibility

# Geometric Brownian Motion (GBM) parameters
last_price = df[df['price_EUR_per_MWh'] > 0]['price_EUR_per_MWh'].iloc[-1]
delta_t = 1  # hourly steps
sigma = hourly_vol
mu = df['log_return'].mean()

# Safety check
if last_price == 0 or np.isnan(last_price):
    raise ValueError("Last known price is invalid (0 or NaN). Check your dataset.")
print("Starting price:", last_price)

# Additive version of GBM-like function
def gbm_additive(mu, sigma, last_price, num_simulations, dt, steps):
    # Initialize price paths
    simulations = np.zeros((steps, num_simulations))
    simulations[0, :] = last_price

    for t in range(1, steps):
        # Generate random step for each simulation
        price_step = (mu - 0.5 * sigma**2) * dt + sigma * np.random.normal(0, np.sqrt(dt), size=num_simulations)
        # Add step to previous price
        simulations[t, :] = simulations[t-1, :] + price_step

    return simulations

# Run additive simulation
simulated_prices_additive = gbm_additive(mu, sigma, last_price, num_simulations, delta_t, forecast_horizon_hours)

# Preview first few steps of one simulation
print("First 10 steps of simulation 0:")
print(simulated_prices_additive[:10, 0])

# forecast index
last_dt = df['datetime'].iloc[-1]
if last_dt.tzinfo is None:
    last_dt = last_dt.tz_localize(time_zone)  # 'Europe/Lisbon'

forecast_index = pd.date_range(
    start=last_dt + pd.Timedelta(hours=1),
    periods=forecast_horizon_hours,
    freq='h'
)

# Plot all simulations
plt.figure(figsize=(16, 6))
for i in range(num_simulations):
    plt.plot(forecast_index, simulated_prices_additive[:, i], color='gray', alpha=0.1)

# Plot mean forecast
#mean_forecast = simulated_prices_additive.mean(axis=1)
#plt.plot(forecast_index, mean_forecast, color='blue', label='Mean Forecast', linewidth=2)

# Plot last known price
plt.scatter(df['datetime'].iloc[-1], last_price, color='red', label='Last Known Price', zorder=5)

# Annotate the red dot with its price value
plt.annotate(f'{last_price:.2f} EUR/MWh',
             xy=(df['datetime'].iloc[-1], last_price),
             xytext=(10, 10),
             textcoords='offset points',
             color='red',
             fontsize=10)

# Set axis limits
plt.xlim(left=df['datetime'].iloc[-1])

# Style: remove grid and spines except bottom/left
for spine in ['top', 'right']:
    plt.gca().spines[spine].set_visible(False)

plt.title('2-Week Energy Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price (EUR/MWh)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Print forecast summary
forecast_summary = pd.DataFrame({
    'datetime': forecast_index,
    'mean_forecast': simulated_prices_additive.mean(axis=1),
    '5th_percentile': np.percentile(simulated_prices_additive, 5, axis=1),
    '95th_percentile': np.percentile(simulated_prices_additive, 95, axis=1)
})
