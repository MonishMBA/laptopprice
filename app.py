import pickle
from flask import Flask, request, render_template
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)


with open('laptop_price_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
    
# Define the home route (renders the form and processes the prediction)
@app.route('/', methods=['GET', 'POST'])
def home():
    predicted_price = None  # Default to no predicted price
    if request.method == 'POST':
        # Extract form data
        company = request.form['Company']
        processor_brand = request.form['Processor_brand']
        processor_suffix = request.form['Processor_suffix']
        memory_type = request.form['Memory Type']
        gpu_category = request.form['Gpu_category']
        os = request.form['OpSys']
        retina_display = int(request.form['Retina Display'])
        ips_panel = int(request.form['IPS Panel'])
        touchscreen = int(request.form['Touchscreen'])
        inches = float(request.form['Inches'])
        clock_speed = float(request.form['Clock_speed(GHz)'])
        ram = float(request.form['RAM(GB)'])
        memory_size = float(request.form['Memory Size (GB)'])
        weight = float(request.form['Weight(kg)'])

        # Create a DataFrame from the form data
        input_data = pd.DataFrame([{
            'Company': company,
            'Inches': inches,
            'Touchscreen': touchscreen,
            'IPS Panel': ips_panel,
            'Retina Display': retina_display,
            'Processor_brand': processor_brand,
            'Processor_suffix': processor_suffix,
            'Clock_speed(GHz)': clock_speed,
            'RAM(GB)': ram,
            'Memory Type': memory_type,
            'Memory Size (GB)': memory_size,
            'Gpu_category': gpu_category,
            'OpSys': os,
            'Weight(kg)': weight
        }])

        predicted_price = round(loaded_model.predict(input_data)[0])

    return render_template('index.html', predicted_price=predicted_price)

# Run the app
if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
