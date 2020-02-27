for factor in 10 20 30 40 50
do
    python base_model.py run --epochs=800 --factors="$factor" --test_inference=0 --file_name="exchange_rate.txt"
    python base_model.py run --epochs=800 --factors="$factor" --test_inference=0 --file_name="electricity.txt"
    python base_model.py run --epochs=800 --factors="$factor" --test_inference=0 --file_name="solar_AL.txt"
    python base_model.py run --epochs=800 --factors="$factor" --test_inference=0 --file_name="traffic.txt"
done
