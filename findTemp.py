from flask import Flask, render_template, request
import requests

app = Flask(__name__)

# 天气查询页面
@app.route('/', methods=['GET', 'POST'])
def weather():
    if request.method == 'POST':
        city = request.form['city']
        weather_data = get_weather(city)
        if weather_data:
            temperature = weather_data['main']['temp']
            humidity = weather_data['main']['humidity']
            description = weather_data['weather'][0]['description']
            return render_template('result.html', city=city, temperature=temperature, humidity=humidity, description=description)
        else:
            error_message = "无法获取天气信息，请确保输入的城市名称正确。"
            return render_template('error.html', error_message=error_message)
    return render_template('weather.html')

# 调用天气API获取天气信息
def get_weather(city):
    api_key = 'your_api_key'  # 替换为您自己的API密钥
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric'
    response = requests.get(url)
    if response.status_code == 200:
        weather_data = response.json()
        return weather_data
    return None

if __name__ == '__main__':
    app.run(debug=True)
