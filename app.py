
from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
from datetime import datetime, timedelta
import pytz

app = Flask(__name__)

# CSV에서 직접 학습
df = pd.read_csv("sample_weather_data.csv")
le_job = LabelEncoder()
le_result = LabelEncoder()
df["공정코드"] = le_job.fit_transform(df["공정"])
df["결과코드"] = le_result.fit_transform(df["결과"])

X = df[["온도", "습도", "풍속", "강수량", "공정코드"]]
y = df["결과코드"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 공정 목록 (웹용)
JOB_OPTIONS = {
    "formwork": "외부비계설치",
    "concrete_floor": "기초타설",
    "interior_paint": "내부 도장",
    "floor_finish": "방통",
    "floor1": "1층 타설",
    "roof": "지붕 타설"
}

def predict(job, temp, humidity, wind, rain):
    try:
        job_label = le_job.transform([job])[0]
        features = np.array([[temp, humidity, wind, rain, job_label]])
        pred = model.predict(features)
        return le_result.inverse_transform(pred)[0]
    except Exception as e:
        return f"에러: {e}"

@app.route("/", methods=["GET", "POST"])
def index():
    today = datetime.now(pytz.timezone("Asia/Seoul")).date()
    result_list = []
    selected_job = request.form.get("job_type", "concrete_floor")
    start = request.form.get("start_date", str(today))
    end = request.form.get("end_date", str(today + timedelta(days=7)))

    if request.method == "POST":
        for i in range(7):
            date = datetime.strptime(start, "%Y-%m-%d") + timedelta(days=i)
            temp = 20 + i
            humidity = 60 + (i % 3) * 10
            wind = 1.5
            rain = 0.0 if i % 2 == 0 else 3.0
            prediction = predict(selected_job, temp, humidity, wind, rain)
            result_list.append({
                "날짜": date.strftime("%Y-%m-%d"),
                "온도": temp,
                "습도": humidity,
                "풍속": wind,
                "강수량": rain,
                "예측결과": prediction
            })

    return render_template("index.html",
        results=result_list,
        job_options=JOB_OPTIONS,
        job_key=selected_job,
        start_date=start,
        end_date=end
    )

if __name__ == "__main__":
    app.run(debug=True)
