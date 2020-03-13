mkdir -p data

echo "START LOAD electricity"
cp ./3rd/multivariate-time-series-data/electricity/electricity.txt.gz ./data/electricity.txt.gz
gzip -d ./data/electricity.txt.gz
mv ./data/electricity.txt ./data/electricity

echo "START LOAD exchange_rate"
cp ./3rd/multivariate-time-series-data/exchange_rate/exchange_rate.txt.gz ./data/exchange_rate.txt.gz
gzip -d ./data/exchange_rate.txt.gz
mv ./data/exchange_rate.txt ./data/exchange_rate

echo "START LOAD solar"
cp ./3rd/multivariate-time-series-data/solar-energy/solar_AL.txt.gz ./data/solar_AL.txt.gz
gzip -d ./data/solar_AL.txt.gz
mv ./data/solar_AL.txt ./data/solar

echo "START LOAD traffic"
cp ./3rd/multivariate-time-series-data/traffic/traffic.txt.gz ./data/traffic.txt.gz
gzip -d ./data/traffic.txt.gz
mv ./data/traffic.txt ./data/traffic
